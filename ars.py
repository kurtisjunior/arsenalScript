from sys import exit
from xml.etree.ElementPath import find
import requests
import json
import re

#curl the site
arsenalSiteData = requests.get('https://www.live-footballontv.com/arsenal-on-tv.html')

# print(type(arsenalSiteData))

#deconde from bytes to string
arsenalSiteDataToString = arsenalSiteData.content.decode("utf-8")

#trim everythin unecessary
trimArsenalSiteData = re.search('(?<=<div class="fixture-group">).*', arsenalSiteDataToString)

#regex on fixture and channel
def returnNextFixture(arsContent):
    findFixture = re.search('(?<=fixture__teams">)([\\s\\S]*?)(?=</div>)', arsContent)
    fixture = findFixture.group(0)

    removeWords = ['Arsenal', 'v']
    for i in removeWords:
        fixture = re.sub(i, '', fixture)

    return fixture.strip()

def returnChannel(arsContent):
    findChannel = re.search('(?<=;">)([\\s\\S]*?)(?=</span>)', arsContent)
    channel = findChannel.group(0)
    channelNoNewLines = channel.replace('\n', '')
    splitChannel = channelNoNewLines.split()

    return(' '.join(splitChannel))

def returnDate(arsContent):
    findDate = re.search('(?<=fixture-date">)([\\s\\S]*?)(?=</div>)', arsContent)
    date = findDate.group(0)
    return date.strip()

def returnTime(arsContent):
    findTime = re.search('(?<=fixture__time">)([\\s\\S]*?)(?=</div>)', arsContent)
    time = findTime.group(0)
    return time.strip()

# def saveCallCount():
#     #post to an endpoint to save count
#     return ""

date = returnDate(trimArsenalSiteData.group(0))
time = returnTime(trimArsenalSiteData.group(0))
fixture = returnNextFixture(trimArsenalSiteData.group(0))
channel = returnChannel(trimArsenalSiteData.group(0))

print(fixture)
print(date, "@", time)
print(channel)
