# datswinlstm_memory



## training phase 1: 长序列

model.memory_bank.requires_grad = True

outputs = model(inputs,memory_x=train_data,phase=1)

outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()

loss_phase_1 = loss(outputs, targets_)

loss_phase_1.backward()

optimizer.step()

## training phase 2: 短序列

model.memory_bank.requires_grad = False

outputs = model(inputs,memory_x=inputs,phase=2)

outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()

loss_phase_2 = loss(outputs, targets_)

loss_phase_2.backward()

optimizer.step()

