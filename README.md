# nanoViT
Vision Transformer (ViT) implementations from scratch


---



## ViT Architecture

<p align="center">
  <img src="images/vit.png" alt="ViT architecture" width="400"/>
</p>




---


## Vision Transformer Model Configuration
```
n_layer    = 4
n_head     = 4
n_embd     = 768 
 
```

---


## Progress


### Step 1 

**Training Configuration**
```
Max Iterations = 100 
Learning Rate  = 3e-4  
Batch Size     = 32  

MNIST Dataset :
image size     = 28 x 28 
image channels = 1
patch size     = 4 x 4
```

<table>
  <tr>
    <td valign="top" width="50%">
      <h4>Training Results</h4>
      <pre>
 0/100   2.1963953971862793
10/100   2.7260947227478027
20/100   2.0315542221069336
30/100   1.3844249248504639
40/100   1.2787163257598877
50/100   0.9120548367500305
60/100   1.3939018249511719
70/100   1.0994621515274048
80/100   0.7976328730583191
90/100   0.8562922477722168
      </pre>
    </td>
    <td valign="top" width="50%">
      <h4>Loss Curve</h4>
      <img src="images/s1.png" alt="Loss curve - Step 1" width="420"/>
    </td>
  </tr>
</table>



---
### Step 2

- Trained on T4 GPU (free Google Colab)

**Techniques**
```
Flash Attention
Dropouts 

torch.compile() : kernel fusion
fused optimizer (AdamW)
```



**Training Configuration**
```
Max Iterations = 1000
Learning Rate  = 3e-4  
Batch Size     = 32  

MNIST Dataset :
image size     = 28 x 28 
image channels = 1
patch size     = 4 x 4
```

<table>
  <tr>
    <td valign="top" width="50%">
      <h4>Training Results</h4>
      <pre>
0/1000  2.2685  131.6884 ms
100/1000  0.8162  88.0437 ms
200/1000  0.9531  85.7632 ms
300/1000  2.0218  89.4675 ms
400/1000  0.7352  87.0714 ms
500/1000  0.4921  83.8625 ms
600/1000  0.7189  87.1720 ms
700/1000  0.7873  88.3372 ms
800/1000  0.9585  88.5544 ms
900/1000  0.7969  86.7596 ms
Time for just one iteration
      </pre>
    </td>
    <td valign="top" width="50%">
      <h4>Loss Curve</h4>
      <img src="images/s2.png" alt="Loss curve - Step 2" width="420"/>
    </td>
  </tr>
</table>

**Validation accuracy**
- 0.8095




---



### Step 3

- Above model for CIFAR10 Trained on T4 GPU



**Training Configuration**
```
Max Iterations = 1000
Learning Rate  = 3e-4  
Batch Size     = 32  

CIFAR10 Dataset :
image size     = 32 x 32 
image channels = 3
patch size     = 8 x 8 
```

<table>
  <tr>
    <td valign="top" width="50%">
      <h4>Training Results</h4>
      <pre>
100/1000  2.3094  40.5657 ms
200/1000  2.1336  37.4801 ms
300/1000  2.3655  38.1184 ms
400/1000  1.8908  39.4473 ms
500/1000  1.8144  38.4216 ms
600/1000  1.7888  39.7465 ms
700/1000  2.0120  39.7189 ms
800/1000  1.7470  40.0531 ms
900/1000  1.9166  40.8957 ms
Time for just one iteration
      </pre>
    </td>
    <td valign="top" width="50%">
      <h4>Loss Curve</h4>
      <img src="images/s3.png" alt="Loss curve - Step 3" width="420"/>
    </td>
  </tr>
</table>

**Validation accuracy**
- 0.2684
- Need to Improve


---




### Step 4 
- Training Techniques
- LR Schedule 
- Weight Decay : AdamW (decoupled Adam)
- Gradient Clipping

**Training Configuration**
```
Max Iterations = 1000 
Batch Size     = 32  

CIFAR10 Dataset :
image size     = 32 x 32 
image channels = 3
patch size     = 8 x 8 
```



<table>
  <tr>
    <td valign="top" width="50%">
      <h4>Training Results</h4>
      <pre>
0/1000  2.3342  23087.0581 ms   norm:7.0133   lr:1.2000e-05
100/1000  2.4822  37.9224 ms   norm:7.2206   lr:7.9537e-04
200/1000  2.5106  40.9629 ms   norm:6.9095   lr:7.5307e-04
300/1000  2.3512  37.1890 ms   norm:7.1988   lr:6.8768e-04
400/1000  1.9129  38.3921 ms   norm:5.2334   lr:6.0628e-04
500/1000  1.8971  36.9473 ms   norm:5.1916   lr:5.1770e-04
600/1000  1.8100  37.8897 ms   norm:3.7129   lr:4.3154e-04
700/1000  1.9884  37.7500 ms   norm:3.8145   lr:3.5713e-04
800/1000  1.8460  40.1011 ms   norm:3.9867   lr:3.0254e-04
900/1000  1.8357  37.7097 ms   norm:3.8326   lr:2.7368e-04
Time for just one Iteration
      </pre>
      <h4>LR Schedule: Warm up + Cosine decay</h4>
      <img src="images/s4__.png" alt="Lrs - Step 3" width="300"/>
    </td>
    <td valign="top" width="50%">
      <h4>Loss Curve</h4>
      <img src="images/s4.png" alt="Loss curve - Step 3" width="400"/>
      <h4>Norms</h4>
      <img src="images/s4_.png" alt="Norms - Step 3" width="400"/>
    </td>
  </tr>
</table>

**Validation accuracy**
```
0.2956 : Batch Size = 32
```

---







## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
