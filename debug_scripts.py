

self.eval()
a = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.memory_inputs[self.buffer.memory_labels == 1].cuda(),with_details=True)[1][-1]), torch.LongTensor(235).fill_(1)))
b = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.abandoned_class_inputs[1][0].cuda(),with_details=True)[1][-1]), torch.LongTensor(265).fill_(1)))
plt.scatter(a[:,0], a[:,1], c=1)
plt.scatter(b[:,0], b[:,1], c=2)
plt.show()


self.eval()
a = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.memory_inputs[self.buffer.memory_labels == 1].cuda(),with_details=True)[1][-1]), torch.LongTensor(226).fill_(1)))
b = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.abandoned_class_inputs[1][0].cuda(),with_details=True)[1][-1]), torch.LongTensor(274).fill_(1)))

plt.scatter(a[:,0], a[:,1], c='red')
plt.scatter(b[:,0], b[:,1], c='green')
plt.show()

a = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.memory_inputs[self.buffer.memory_labels == 1].cuda(),with_details=True)[1][-1]), torch.LongTensor(to_numpy(self.buffer.memory_labels == 1).sum().astype(np.int)).fill_(1)))
b = to_numpy(self.prototypes.transform(normalize(self.forward(self.buffer.abandoned_class_inputs[1][0].cuda(),with_details=True)[1][-1]), torch.LongTensor(len(self.buffer.abandoned_class_inputs[1][0])).fill_(1)))
plt.scatter(a[:,0], a[:,1], c='red')
plt.scatter(b[:,0], b[:,1], c='green')
plt.show()


to_numpy(self.task_encoder_state_dict[0]['layer4.0.bn1.running_var'] - self.model.encoder.state_dict()['layer4.0.bn1.running_var'])


self.model.task_encoder_state_dict['layer_1_1.bn1.bn.running_var']
