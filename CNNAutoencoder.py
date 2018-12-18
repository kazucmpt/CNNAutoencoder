import cv2
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L 
import chainer.functions as F 
from chainer import cuda, Variable, optimizers, Chain, datasets

#Define CNN Autoencoder
class CNNAutoencoder(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(None,16,5)
			self.dcnv = L.Deconvolution2D(None,3,5)

	def __call__(self,x):
		h = F.relu(self.conv(x))
		h = F.relu(self.dcnv(h))
		
		return h

def main():
	max_epoch = 51
	batchsize = 64

	data, _ = chainer.datasets.get_cifar10()
	train_input = np.zeros((len(data),3,32,32), dtype=np.float32)
	for i in range(0,len(data)):
		train_input[i] = data[i][0]
	train_input = Variable(train_input)

	model = CNNAutoencoder()
	model.to_gpu(0)
	optimizer = optimizers.SGD(lr=0.001).setup(model)

	
	N = len(train_input)
	perm = np.random.permutation(N)
	for epoch in range(max_epoch):
		for i in range(0,N,batchsize):
			train_input_batch = train_input[perm[i:i + batchsize]]
			train_input_batch.to_gpu(0)
			t = model(train_input_batch)
			loss = F.mean_squared_error(t,train_input_batch)
			model.cleargrads()
			loss.backward()
			optimizer.update()
	
		print("epoch:", epoch, "loss:",loss.data)
	
		input_img = Variable(np.array([train_input[0].array],dtype=np.float32))
		input_img.to_gpu(0)
		output_img = model(input_img)
		output_img.to_cpu()
		output_img = (output_img.array*255)/255
		plt.imshow(output_img[0].transpose(1,2,0))
		plt.savefig("img/epoch{}.png".format(epoch))
		plt.close()	

	input_img = train_input[0].array
	plt.imshow(input_img.transpose(1,2,0))
	plt.savefig("img/GT.png")
	plt.show()


if __name__ == "__main__":
	main()
