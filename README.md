# End-to-end diarization loss
This is the implement of PIT loss, FastPIT loss and OPTM loss for end-to-end diarization training. Acknowledge @tdedecko for the implement of Hungarian algorithm at https://github.com/tdedecko/hungarian-algorithm. 

### Prerequisites
```
Python >= 3.7
Pytorch >= 1.3.1
numpy >= 1.15.1
```

## Simple example:
The loss directory is regarded as the python package. Usage:
```
import torch
from loss import PITLoss, FastPITLoss, OPTMLoss

B, T, n_class = 1, 4, 4     # batch_size * nframe * n_speaker
device = 'cpu'
pred = torch.rand(B, T, n_class).to(device)
label = torch.randint(0, 2, (B, T, n_class)).float().to(device)

criterion1 = PITLoss()
criterion2 = FastPITLoss()
criterion3 = OPTMLoss()
loss1, assigned_label1 = criterion1(pred, label)
loss2, assigned_label2 = criterion2(pred, label)
loss3, assigned_label3 = criterion3(pred, label)
print(loss1, loss2, loss3)
print(assigned_label1)
print(assigned_label2)
print(assigned_label3)
```

## Citation
```
Q. Lin et al.: Optimal Mapping Loss: A Faster Loss for End-to-End Speaker Diarization, Odyssey 2020.
```