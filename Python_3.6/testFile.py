from ANN_class import ANN
import mnistHandwriting
import matplotlib.pyplot as plt
import numpy as np
 
print("Loading Data")
T = mnistHandwriting.MNISTexample(0,60000)
Test = mnistHandwriting.MNISTexample(0,1000,bTrain=True)
ANN_network = ANN([784,20,25,10])
ANN_network.mini_batch_training(T, 300, 3,250)
plt.plot(ANN_network.cost_result)       
plt.plot(ANN_network.accuracy_result,'r-')
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
print(correct/total)
plt.show()
