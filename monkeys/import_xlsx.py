import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from data_interface.import_export import import_data_xlsx


if __name__ == "__main__":

    import_data_xlsx(data_files=('data_GH.xlsx', 'data.xlsx'),
                     starting_dates=("2017-03-01", "2020-02-18"))
