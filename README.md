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
### Step 1 

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

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
