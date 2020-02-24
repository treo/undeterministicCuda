To run:

```
git clone https://github.com/treo/undeterministicCuda.git
cd undeterministicCuda
chmod +x run.sh
./run.sh
```

# Model

```
===================================================================================================================
LayerName (LayerType)          nIn,nOut    TotalParams   ParamsShape                                               
===================================================================================================================
layer0 (ConvolutionLayer)      3,16        432           W:{16,3,3,3}                                              
layer1 (BatchNormalization)    16,16       64            gamma:{1,16}, beta:{1,16}, mean:{1,16}, var:{1,16}        
layer2 (ActivationLayer)       -,-         0             -                                                         
layer3 (SubsamplingLayer)      -,-         0             -                                                         
layer4 (ConvolutionLayer)      16,32       4.608         W:{32,16,3,3}                                             
layer5 (BatchNormalization)    32,32       128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, var:{1,32}        
layer6 (ActivationLayer)       -,-         0             -                                                         
layer7 (SubsamplingLayer)      -,-         0             -                                                         
layer8 (ConvolutionLayer)      32,64       18.432        W:{64,32,3,3}                                             
layer9 (BatchNormalization)    64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}        
layer10 (ActivationLayer)      -,-         0             -                                                         
layer11 (SubsamplingLayer)     -,-         0             -                                                         
layer12 (ConvolutionLayer)     64,128      73.728        W:{128,64,3,3}                                            
layer13 (BatchNormalization)   128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}    
layer14 (ActivationLayer)      -,-         0             -                                                         
layer15 (SubsamplingLayer)     -,-         0             -                                                         
layer16 (ConvolutionLayer)     128,256     294.912       W:{256,128,3,3}                                           
layer17 (BatchNormalization)   256,256     1.024         gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}    
layer18 (ActivationLayer)      -,-         0             -                                                         
layer19 (SubsamplingLayer)     -,-         0             -                                                         
layer20 (ConvolutionLayer)     256,512     1.179.648     W:{512,256,3,3}                                           
layer21 (BatchNormalization)   512,512     2.048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}    
layer22 (ActivationLayer)      -,-         0             -                                                         
layer23 (SubsamplingLayer)     -,-         0             -                                                         
layer24 (ConvolutionLayer)     512,1024    4.718.592     W:{1024,512,3,3}                                          
layer25 (BatchNormalization)   1024,1024   4.096         gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}
layer26 (ActivationLayer)      -,-         0             -                                                         
layer27 (ConvolutionLayer)     1024,1024   9.437.184     W:{1024,1024,3,3}                                         
layer28 (BatchNormalization)   1024,1024   4.096         gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}
layer29 (ActivationLayer)      -,-         0             -                                                         
layer30 (ConvolutionLayer)     1024,30     30.750        b:{1,30}, W:{30,1024,1,1}                                 
layer31 (Yolo2OutputLayer)     -,-         0             -                                                         
-------------------------------------------------------------------------------------------------------------------
            Total Parameters:  15.770.510
        Trainable Parameters:  15.770.510
           Frozen Parameters:  0
===================================================================================================================

```