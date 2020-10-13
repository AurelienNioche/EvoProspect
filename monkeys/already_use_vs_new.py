import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from data_interface.models import Data
import datetime

total_old = 0
total_n = 0

for m in 'Havane', 'Gladys':
    entries = Data.objects.filter(monkey=m)

    n = entries.count()
    total_n += n

    last_rec = datetime.datetime.fromisoformat("2018-02-23")
    old = entries.filter(date__lte=last_rec).count()
    total_old += old

    print(m)
    print("n", n)
    print("old", old)
    print(f"reused: {old/n*100:.2f}%")
    print()

print("both monkeys")
print("total n", total_n)
print("total old", total_old)
print(f"reused: {total_old/total_n*100:.2f}%")
