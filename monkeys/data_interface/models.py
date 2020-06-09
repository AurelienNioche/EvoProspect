from django.db import models


class Data(models.Model):

    monkey = models.TextField(blank=True, null=True)
    date = models.DateTimeField(blank=True, null=True)

    c = models.IntegerField(blank=True, null=True)

    p0 = models.FloatField(blank=True, null=True)
    x0 = models.IntegerField(blank=True, null=True)
    p1 = models.FloatField(blank=True, null=True)
    x1 = models.IntegerField(blank=True, null=True)

    is_gain = models.BooleanField(default=None, null=True)
    is_loss = models.BooleanField(default=None, null=True)

    is_control = models.BooleanField(default=None, null=True)
    is_same_p = models.BooleanField(default=None, null=True)
    is_same_x = models.BooleanField(default=None, null=True)
    is_best_left = models.BooleanField(default=None, null=True)
    is_best_right = models.BooleanField(default=None, null=True)

    is_risky = models.BooleanField(default=None, null=True)
    is_risky_left = models.BooleanField(default=None, null=True)
    is_risky_right = models.BooleanField(default=None, null=True)

    choose_risky = models.BooleanField(default=None, null=True)
    choose_best = models.BooleanField(default=None, null=True)

    pair_id = models.IntegerField(blank=True, null=True)
    is_reversed = models.BooleanField(default=None, null=True)

    class Meta:
        db_table = 'data'
