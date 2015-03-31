#!/bin/pyspark
import numpy as np


basepath = '/rahul_extra/Machine Learning/HW3Data/'


def set_feature(view,f):
  x = float(view.split(' ')[2])
  return ((view.split(' ')[0],str(f)+".txt",view.split(' ')[1]),x)

def readfile():
	fr = range(6,23)
	tfile = sc.union([sc.textFile(basepath+str(f)+".txt")
		                .map(lambda view: set_feature(view,f)) 
		                .reduceByKey(lambda a, b: a+b)
		                for f in fr])
	return tfile


def create_feature(tfile):
	collection = tfile.collect()
	d = {}
	for data in collection:
		if not d.has_key((data[0][0],data[0][2])):
			x = np.zeros(19)
			x[0] = 1
			fno = str(data[0][1]).split('.')[0]
			x[int(fno)-5] = data[1]
			d[(data[0][0],data[0][2])] = x
		else:
			x = d[(data[0][0],data[0][2])]
			fno = str(data[0][1]).split('.')[0]
			x[int(fno)-5] = data[1]
			d[(data[0][0],data[0][2])] = x
	return d

def get_target_info():
	tfile = sc.textFile(basepath+'23.txt')
	collection = tfile.map(lambda view: ((view.split(' ')[0],view.split(' ')[1]),float(view.split(' ')[2]))).reduceByKey(lambda a,b: a+b).collect()
	target = {}
	for data in collection:
		target[data[0]] = data[1]
	return target


def main():
	tfile = readfile()
	dcollec = create_feature(tfile)
	print "the aggregated data in 6-22 collections is",dcollec
	dtarget = get_target_info()
	print "the data in 23rd file is", dtarget
	for key in dtarget:
		if dcollec.has_key(key):
			x = dcollec[key]
			x[18] = dtarget[key]
			dcollec[key] = x
		else:
			print "the key",key,"is not the dcollection... "
	print "the update aggregated data in dcollec is ",dcollec



'''
sc.textFile(base+'/'+str(fno)+'.txt')
tfile_rdd = tfile.map(lambda view: set_feature(view))
'''