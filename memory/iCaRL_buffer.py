
from memory.base_buffer import *

def random_retrieve(buffer, n_retrieve):
    new_index = buffer.current_index + n_retrieve
    indices = np.arange(buffer.current_index, new_index)
    indices = indices % buffer.memory_labels.shape[0]
    buffer.current_index = new_index
    if 0 in indices:
        buffer.shuffer()
    indices = buffer.random_idx[indices]
    x = buffer.memory_inputs[indices]
    y = buffer.memory_labels[indices]
    t = buffer.memory_tasks[indices]
    f = buffer.memory_logits[indices]
    return x.cuda(), y.cuda().long(), t.cuda().long(), f.cuda()


def iCaRL_update(buffer, x, y, t, features, model):
    mem_inputs = []
    mem_labels = []
    mem_tasks = []
    mem_logits = []
    class_samples_upper = int(buffer.n_memories / ((t + 1) * 10)) #(buffer.n_classes / buffer.n_tasks)

    for m_cls in buffer.class_n_samples.keys():
        m_cls_idx = torch.nonzero(buffer.memory_labels == m_cls).squeeze()[:class_samples_upper]
        m_cls_x = buffer.memory_inputs[m_cls_idx]
        m_cls_y = buffer.memory_labels[m_cls_idx]
        m_cls_t = buffer.memory_tasks[m_cls_idx]
        mem_inputs.append(m_cls_x)
        mem_labels.append(m_cls_y)
        mem_tasks.append(m_cls_t)
        m_features = model.encoder_feature(m_cls_x.cuda())
        m_logits = model(m_cls_x.cuda())
        mem_logits.append(m_logits.cpu())
        buffer.prototypes[m_cls] = m_features.mean(0)
        buffer.class_n_samples[m_cls] = class_samples_upper

    current_classes = set(to_numpy(y))
    for t_cls in current_classes:
        t_cls_idx = torch.nonzero(y == t_cls).squeeze()
        mu_feature = features[t_cls_idx].mean(0)
        t_cls_idx_list = to_numpy(t_cls_idx).tolist()
        selected_idx = []
        # for i in range(class_samples_upper):
        #     min_idx = -1
        #     min_dis = 1e10
        #     selected_mu_feature = features[selected_idx].sum(0)
        #
        #     for idx in t_cls_idx_list:
        #         f = features[idx]
        #         dis = torch.linalg.norm((f + selected_mu_feature) / (1 + len(selected_idx)) - mu_feature).cpu()
        #         if dis < min_dis:
        #             min_idx = idx
        #             min_dis = dis
        #     selected_idx.append(min_idx)
        #     t_cls_idx_list.remove(min_idx)
        selected_idx = np.random.choice(t_cls_idx_list, class_samples_upper, replace=False)

        t_cls_x = x[selected_idx]
        t_cls_y = y[selected_idx]
        mem_inputs.append(t_cls_x)
        mem_labels.append(t_cls_y)
        mem_tasks.append(torch.zeros_like(t_cls_y).fill_(t))
        mem_logits.append(model(t_cls_x.cuda()).cpu())
        buffer.prototypes[t_cls] = features[selected_idx].mean(0)
        buffer.class_n_samples[t_cls] = class_samples_upper

    buffer.prototypes = unit_vector(buffer.prototypes)
    buffer.memory_inputs = torch.cat(mem_inputs, dim=0)
    buffer.memory_labels = torch.cat(mem_labels, dim=0)
    buffer.memory_tasks = torch.cat(mem_tasks, dim=0)
    buffer.memory_logits = torch.cat(mem_logits, dim=0)
    buffer.shuffer()



class iCaRL_buffer(Buffer):
    def __init__(self, n_tasks, n_memories, n_classes, input_dims):
        super(iCaRL_buffer, self).__init__(n_tasks, n_memories, n_classes)
        self.prototypes = torch.FloatTensor(n_classes, input_dims).cuda()
        self.memory_logits = torch.FloatTensor()


