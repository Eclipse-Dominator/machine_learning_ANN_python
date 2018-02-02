from ANN_class import ANN
import mnistHandwriting
import matplotlib.pyplot as plt
import numpy as np
 
print "Loading Data"
T = mnistHandwriting.MNISTexample(0,10000)
Test = mnistHandwriting.MNISTexample(0,10000,bTrain=True)
ANN_network = ANN([784,30,30,10])
ANN_network.mini_batch_training(T, 50, 0.3,50)
plt.plot(ANN_network.cost_result)
plt.plot(ANN_network.accuracy_result,'r-')
plt.show()
correct = 0.0
total = 0.0
wrong = []
for i in Test:
	network_result = ANN_network.propagate_result(i[0])
	if np.argmax(network_result) == np.argmax(i[1]):
		correct += 1
		#print network_result
	else:
		wrong.append(i)
	total += 1
#mnistHandwriting.writeMNISTimage(wrong)
print correct/total
