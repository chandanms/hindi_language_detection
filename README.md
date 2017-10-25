# Detection of hindi language in natural images using machine learning.
Detecting hindi language from natural images.


Guide to use :
  preprocess.py - The image is preprocessed to detect objects and texts in the image and stores them seperately in the existing
  folder. The image to be tested shouldbe named 'test.jpg'. 
  
  text_detection.py - This is the machine learning model which uses concolutional neural network to train various preprocessed      
  images obtained from the preprocess.py. The model is then stored to predict the images given. It takes two folders namely 
  train and validation and trains on it to forma model. The model is then saved to predict on images.
  
  text_prediction.py - Given a segment of a image, it predicts based on the training given ion the previous script.
  
  data.zip - The datas scraped from various preprocessed image. It is divided into train and validation. The data is limited due to 
  poor computational ability. It is advised to scrape more naturalimages and give it to preprocessing. 
  
To summarize, the image given must first be preprocessed which gives various segmented images which is then given to text_prediction.py script which predicts if the segment is hindi(1) or not(0).

Please see : Due to lack of computational power of my age old laptop, I could not train for many images which resulted in fall of accuracy. Also, clearer the image, better the results.
  
  
  
  
  
