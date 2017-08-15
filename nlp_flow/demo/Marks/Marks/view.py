# coding=utf-8

from __future__ import unicode_literals

from django.http import HttpResponse

from django.shortcuts import render

import requests
import os
import json
from collections import OrderedDict

import sys

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)


def extractData(data):
    # 调用当前路径
    # p = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'post_res.json')
    # f = open(p)
    # data = json.load(f)

    areas = dict()  # 每个区域为一组(宫体,宫颈...)

    empty_area = []

    for k in data.iterkeys():
        if len(data[k].keys()) == 1:
            empty_area.append(k)
        else:
            for k1, v1 in data[k].iteritems():
                if k1 != "range":
                    if u'子区域' in v1:
                        splitArea = v1[u'子区域'].split('_')
                        data[k][k1][u'子区域'] = splitArea
                    if u'描述' in v1:
                        splitDesc = v1[u'描述'].split('_')
                        data[k][k1][u'描述'] = splitDesc

    # f.close()

    return data, empty_area


def mainPage(request):
    # areas = extractData()

    # render第3个参数向网页传值
    return render(request, 'main.html')


def postData(request):
    areas = dict()
    rate = 0
    if request.method == 'POST':
        data = {
            'mri_exam': request.POST.get("msg")
        }

        # json
        d = requests.post('http://127.0.0.1:5002/extract-structural-info', data=data)
        areas, empty_area = extractData(d.json())
        for k, v in areas.iteritems():
            print k
            for k1, v1 in v.iteritems():
                print k1
                for k2, v2 in v1.iteritems():
                    print k2, v2

        # 发病概率
        rate = requests.post('http://127.0.0.1:5002/mri-outcome-score', data=data)

        print "rate=", rate.text

        return HttpResponse(json.dumps({
            'msg': request.POST.get("msg"),
            'areas': areas,
            'empty_area': empty_area,
            'rate': rate.text}))
    return render(request, 'main.html', {
        'areas': areas,
        'rate': rate
    })

def test(request):
    return render(request,"1.html")