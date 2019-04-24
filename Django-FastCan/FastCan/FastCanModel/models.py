from django.db import models
import os

def getFilePath(instance, filename):
    return os.path.join(instance.sessionKey, filename)

# Create your models here.

class Profile(models.Model):
    sessionKey = models.CharField(max_length=40, primary_key=True)
    createdAt = models.DateTimeField(auto_now=True)
    document = models.FileField(upload_to=getFilePath, null=True)
    trainData = models.FileField(upload_to=getFilePath, null=True)

# User Creation

'''
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    document = models.FileField(upload_to='documents/')

@receiver(post_save, sender=User)
def updateProfile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()
'''