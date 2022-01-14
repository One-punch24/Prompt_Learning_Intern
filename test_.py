str="Epoch 8, global_step 25500 average loss: 13.725779510498047 lr: 1.5141430948419303e-06"

str_=float((str.split("loss:")[1].split("lr:"))[0])
print(str_)