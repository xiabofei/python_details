#encoding=utf8
"""
parse xml file from jiahe CDR
"""
from lxml import etree
import sys
import os
from fnmatch import fnmatch


def dfs(root):
    if root.tag=='section': print root.attrib['name'].encode('utf-8')
    # step1. parse 'text' element under a 'section' element
    text = []
    for child in root.findall('text'):
        child.text!=None and text.append(child.text)
    if len(text)>0:
        print "text content:"
        print " ".join(text).encode('utf-8')
    # step2. parse 'fieldelem' element under a 'section' element
    fieldelem = []
    for child in root.findall('fieldelem'):
        child.text!=None and fieldelem.append(child.attrib['name']+":"+child.text)
    if len(fieldelem)>0:
        print "fieldelem content:"
        print "\n".join(fieldelem).encode('utf-8') 
    # step3. dfs next level
    child_node = root.getchildren()
    if len(child_node) == 0: return
    for child in child_node: dfs(child)

if __name__ == '__main__':
    for f in os.listdir('.'):
        if fnmatch(f,'*.xml'):
            print f
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(f, parser)
            root = tree.getroot()
            dfs(root)
