from django.shortcuts import render
from django.shortcuts import redirect
from FastCanModel.form import ProfileForm
from FastCanModel.models import Profile
from FastTextWeb import TextClassifier
from FastTextWeb import LoadModel
from django.conf import settings
from FastTextWeb import TrainModel
from FastTextWeb import SearchUser
import json
import os

# Create your views here.

def LoadClassifier(request):

    user = Profile.objects.get(sessionKey=str(request.session.session_key))

    if user.trainData:
        TrainModel(user.trainData.file.name)

    if user.document:
        return redirect('/table/')

    return redirect('/result/')

def ShowUser(request):

    ResultList = SearchUser(10)

    return render(request, 'table.html', {'ResultList': ResultList})

def ShowTable(request):

    user = Profile.objects.get(sessionKey=str(request.session.session_key))
    model_path = os.path.join(settings.MEDIA_ROOT, str(request.session.session_key))
    classifier = LoadModel(user.trainData, model_path)
    ResultList = []

    with open(os.path.join(settings.MEDIA_ROOT,str(user.document)), 'r', encoding='utf-16') as f:
        content = f.readlines()
        ResultList = BatchClassifier(content, classifier)

    return render(request, 'table.html', {'ResultList': ResultList})

def Classifier(request):

    if request.POST.get('sentence'):

        user = Profile.objects.get(sessionKey=str(request.session.session_key))
        result = request.POST.get('sentence')
        model_path = os.path.join(settings.MEDIA_ROOT, str(request.session.session_key))
        classifier = LoadModel(user.trainData, model_path)
        ResultList = []

        for label in TextClassifier(result, classifier):
            for s in label:
                result = result + ' [' + s + ']'

        if 'ResultList' in request.session:
            ResultList = request.session['ResultList']
            ResultList.append(result)
            request.session['ResultList'] = ResultList
        else:
            request.session['ResultList'] = [result]

        return render(request, 'result.html', {'ResultList' : request.session['ResultList']})

    else:

        return render(request, 'result.html')

def Clear(request):
    
    if 'ResultList' in request.session:
        del request.session["ResultList"]
    return render(request, 'result.html')

def SaveForm(request):

    if request.method == 'POST':

        if not request.session.session_key:
            request.session.cycle_key()

        form = ProfileForm(request.POST, request.FILES)

        if form.is_valid():

            try:
                model = Profile.objects.get(sessionKey=str(request.session.session_key))
                model.document = form.cleaned_data['document']
                model.trainData = form.cleaned_data['trainData']
                model.save()
            except Profile.DoesNotExist:
                model = Profile.objects.create(sessionKey=str(request.session.session_key))
                model.document = form.cleaned_data['document']
                model.trainData = form.cleaned_data['trainData']
                model.save()

            return redirect('/load/')

    else:

        form = ProfileForm()

    return render(request, 'home.html', {'form': form})