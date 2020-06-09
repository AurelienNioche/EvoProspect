from django.contrib import admin

from . models import Data


# Register your models here.
class DataAdmin(admin.ModelAdmin):
    list_display = [f.name for f in Data._meta.get_fields()]


admin.site.register(Data, DataAdmin)
