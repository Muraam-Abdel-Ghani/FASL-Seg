# FASL-Seg-v1
This repo contains the model creation and training scripts for the proposed FASL-Seg-v1 model.

# Description
There are two main files for the creation of this model. The LLFDecoder file (llf_decoder.py) extracts the SegFormer encoder outputs and passes them through the two feature processing streams, LLFP and HLFP. The outputs of the streams are concatenated and passed through the first two convolution layers to compress the channel sizes. 

The output of the LLFDecoder is collected in the fasl_seg_model.py file. Two more convolution layers are applied for channel compression and mixing, the feature map is interpolated to the image size (512x512) and finally passed through the final convolution layer where the channel size is the number of classes.

In train_aug_fasl2_seg.py is the training code for FASL-Seg, where the model is instantiated and the dataloader is created, and can be used to train the model on customized datasets. Please be sure to change all paths and directories to match your local filesystem paths, including where you want to store your model weights and checkpoints. By default, both the model_state_dict and optimizer_state_dict are saved in the model weights file. Feel free to change this implementation to save only model weights: 
```
# Change the torch.save from this ...
 torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()} ,
                   f"path/to/file.pth")
# ... to this:
 torch.save(model.state_dict(),
                   f"path/to/file.pth")

# If saving both model and optimizer state dictionaries, load each separately as follows:
checkpoint = torch.load(f"path/to/file.pth")

model = FASLSegModel(num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# If saving only model_state_dict(), load your weights using:
model.load_state_dict(torch.load(f"path/to/file.pth"))
```

The implementations of the metrics used for training and testing are present under the metrics directory, namely the mean Intersection over Union (mIoU) and Dice similarity coefficient (Dice). The code for the dataset used for the dataloader is in the dataset.py file.

## Citation
If you are interested to build on this code or use it in any research, make sure to cite our paper in any publications:
```
Citation once published:

```
