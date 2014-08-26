import SocketServer, time, socket, re, os
import matplotlib.pyplot as plt
import numpy as np
import caffe

# Set your GPU mode
# If you want to use GPU for parellel calculation, set use_gpu as True
use_gpu = False

# Set oversample option
oversample = True

caffe_root = 'caffe/'  
MODEL_FILE = 'caffe/examples/imagenet/imagenet_deploy.prototxt'
PRETRAINED = 'caffe/examples/imagenet/caffe_reference_imagenet_model'
LABEL_FILE = 'caffe/data/ilsvrc12/synset_words.txt'
PATTERN = '<id>(.*?)</id>'  #regular expression, used to get id


# Load all labels, return a list of labels
def loadLabel():
    labels=[]
    for line in open(LABEL_FILE):
        l = line.split(',')
        l[0] = ' '.join( l[0].split()[1:] )
        labels.append(l)
    return labels

# CNN_Classify, return a list of possible result labels
def CNN_Classify(imageFile):
    labels = loadLabel()
    
    # Initialize a CNN
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255)

    net.set_phase_test()

    # Set GPU Mode
    if use_gpu:
       net.set_mode_gpu()
    else:
       net.set_mode_cpu()
    input_image = caffe.io.load_image(imageFile) # defined in io.py

    # Set oversample Mode
    if oversample:     
       prediction = net.predict([input_image])
    else:
       prediction = net.predict([input_image], oversample = False)
    
    # Print label in the terminal
    label = labels[prediction[0].argmax(axis=0)]
    print label

    return label

class MyServer(SocketServer.BaseRequestHandler):   
  
    def handle(self):   
        print 'Connected from', self.client_address   
           
        while True:   
            # Receive data from the client
            receivedData = self.request.recv(1024)

            if not receivedData:   
                continue    

            # Things to do with receiving pic from the client
            elif receivedData.startswith('<id>'):
                usrID = re.match(PATTERN, receivedData)
                fileName = usrID.group(1) + '.png'
                f = open(fileName, 'wb')
                count = 0;
                while True:
                    data = self.request.recv(8192)
                    print 'package: ', count
                    count = count + 1
                    if data.find('<END OF FILE>') >= 0:
                        data = data[:-13]
                        f.write(data)
                        print 'finished'
                        break
                    f.write(data)

                f.flush()
                f.close()

                # Get the result from CNN
                result = CNN_Classify(fileName)

                # Send the best reuslt (only one) to the client
                self.request.sendall(result[0])

                # Delete the temp pic been uploaded to the server
                os.remove(fileName)

            elif receivedData.startswith('bye'):   
                break  
  
        self.request.close()   
           
        print 'Disconnected from', self.client_address   
        print  
  
if __name__ == '__main__':   
    print 'Server is started\nwaiting for connection...\n'

    #set your own server ip and port
    srv = SocketServer.ThreadingTCPServer(('127.0.0.1', 10000), MyServer)   
    srv.serve_forever() 
