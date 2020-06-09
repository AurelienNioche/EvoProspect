import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import pytz
import xlsxwriter

from data_interface.models import Data
from parameters.parameters import SOURCE_FOLDER, EXPORT_FOLDER


class Importer:

    BULK_LIMIT = 10000
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, data_files, starting_dates):

        self.starting_dates = [self._get_date(str_d)
                               for str_d in starting_dates]
        self.data_files = data_files

        self.filtered_entries = []
        self.counter = 0

        self.lottery_types = {}

    def _run(self):

        print("Deleting previous entries...", end=" ", flush=True)
        Data.objects.all().delete()
        print("Done!")

        for data_file, starting_date in zip(self.data_files,
                                            self.starting_dates):

            self._import_data_file(data_file=data_file,
                                   starting_date=starting_date)

    def _import_data_file(self, data_file, starting_date):

        data_path = os.path.join(SOURCE_FOLDER, data_file)

        print(f"Reading from '{data_path}'...", end=" ", flush=True)
        df = pd.read_excel(data_path)
        print("Done!")
        entries = df.to_dict('records')

        print("Preprocessing and writing the data...", end=" ", flush=True)

        for i, entry_dic in enumerate(entries):
            self._import_entry(entry_xlsx=entry_dic,
                               starting_date=starting_date)

        print("Done!")

    def _write_in_db(self):

        Data.objects.bulk_create(self.filtered_entries)
        self.filtered_entries = []
        self.counter = 0

    def _import_entry(self, entry_xlsx, starting_date):

        if not self._is_complete(entry_xlsx):
            return

        date = self._get_date(entry_xlsx['date'])
        if date < starting_date:
            return

        if entry_xlsx['error'] != "None":
            return

        entry_content = {
            "p0": entry_xlsx["stim_left_p"],
            "p1": entry_xlsx["stim_right_p"],
            "x0": entry_xlsx["stim_left_x0"],
            "x1": entry_xlsx["stim_right_x0"],
            "date": date,
            "c": entry_xlsx["choice"],
            "monkey": entry_xlsx["monkey"]
        }

        new_db_entry = self._create_db_entry(entry_content)
        if new_db_entry is None:
            return

        self._add_to_filtered_entries(new_db_entry)

    def _create_db_entry(self, entry_content):

        e = Data(**entry_content)

        e.is_gain = e.x0 > 0 and e.x1 > 0
        e.is_loss = e.x0 < 0 and e.x1 < 0

        if not (e.is_gain or e.is_loss):
            return

        e.is_risky_left = \
            np.abs(e.x0) > np.abs(e.x1) and e.p0 < e.p1
        e.is_risky_right = \
            np.abs(e.x0) < np.abs(e.x1) and e.p0 > e.p1

        e.is_risky = e.is_risky_left or e.is_risky_right

        e.is_same_p = e.p0 == e.p1
        e.is_same_x = e.x0 == e.x1
        e.is_best_left = \
            (e.x0 > e.x1 and e.is_same_p) \
            or (e.p0 > e.p1 and e.is_same_x and e.is_gain) \
            or (e.p0 < e.p1 and e.is_same_x and e.is_loss)
        e.is_best_right = \
            (e.is_same_p or e.is_same_x) and not e.is_best_left

        e.is_control = e.is_best_left or e.is_best_right

        if not (e.is_control or e.is_risky):
            return

        if (e.is_risky and e.is_risky_left) \
                or (e.is_control and e.is_best_left):
            pair = (e.p0, e.x0, e.p1, e.x1)
            e.is_reversed = False
        else:
            pair = (e.p1, e.x1, e.p0, e.x0)
            e.is_reversed = True
        try:
            e.pair_id = self.lottery_types[pair]
        except KeyError:
            e.pair_id = len(self.lottery_types)
            self.lottery_types[pair] = e.pair_id

        e.choose_best = (e.is_best_left and e.c == 0) \
            or (e.is_best_right and e.c == 1)

        e.choose_risky = (e.is_risky_left and e.c == 0) \
            or (e.is_risky_right and e.c == 1)

        return e

    def _get_date(self, string_date):
        string_date = string_date.replace('None', '').replace('_', '-')
        return datetime.strptime(string_date, self.DATE_FORMAT)\
            .astimezone(pytz.UTC)

    @staticmethod
    def _is_complete(entry_xlsx):

        complete = True
        for k, v in entry_xlsx.items():
            if str(v) == '' or (type(v) in (float, int) and np.isnan(v)):
                complete = False
                msg = f"I will ignore line id={entry_xlsx['id']} " \
                      f"(missing data or wrong format for column '{k}')"
                warnings.warn(msg)
                break
        return complete

    def _add_to_filtered_entries(self, new_entry):

        self.filtered_entries.append(new_entry)
        self.counter += 1
        if self.counter > self.BULK_LIMIT:
            self._write_in_db()

    @classmethod
    def import_data_xlsx(cls, data_files, starting_dates):

        imp = cls(data_files=data_files, starting_dates=starting_dates)
        imp._run()

    @classmethod
    def export_as_xlsx(cls, file_name='formatted_data.xlsx'):

        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        file_path = os.path.join(EXPORT_FOLDER, file_name)

        workbook = xlsxwriter.Workbook(file_path, {'remove_timezone': True})
        worksheet = workbook.add_worksheet()

        # Write column headers
        col = [f.name for f in Data._meta.get_fields()]
        for j, c in enumerate(col):
            worksheet.write(0, j, c)

        data = Data.objects.values_list(*col)

        for i, d in enumerate(data):
            for j in range(len(col)):

                if j == col.index('date'):
                    entry = d[j].strftime(cls.DATE_FORMAT)

                else:
                    entry = d[j]

                worksheet.write(i + 1, j, entry)

        workbook.close()

        print(f"Data exported in file '{file_path}'")


def import_data_xlsx(data_files, starting_dates, ):
    Importer.import_data_xlsx(data_files=data_files,
                              starting_dates=starting_dates)


def export_as_xlsx():

    Importer.export_as_xlsx('formatted_data.xlsx')
