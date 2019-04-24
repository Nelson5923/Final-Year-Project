from django import forms

# User Creation

class ProfileForm(forms.Form):
    document = forms.FileField(label='Upload your sentence with .txt if any (Optional)', required=False)
    trainData = forms.FileField(label='Upload your train data if any (Optional)', required=False)

'''
class ProfileForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', )
'''

