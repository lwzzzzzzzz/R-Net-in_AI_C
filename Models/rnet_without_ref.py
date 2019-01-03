import tensorflow as tf
import math


class R_NET:
    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal([dim]), name=name)

    def random_scalar(self, name=None):
        return tf.Variable(0.0, name=name)

    def DropoutWrappedGRUCell(self, hidden_size, in_keep_prob, name=None):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def __init__(self, options):
        with tf.device('/cpu:0'):
            self.options = options

            # answer
            self.W_ans2_1 = self.random_weight(options['emb_dim'], 1, name='W_for_pre_answer')
            # Weights	按论文的公式
            self.W_uQ = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uQ')
            self.W_uP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uP')
            self.W_vP = self.random_weight(options['state_size'], options['state_size'], name='W_vP')
            self.W_g_QP = self.random_weight(4 * options['state_size'], 4 * options['state_size'], name='W_g_QP')
            self.W_smP1 = self.random_weight(options['state_size'], options['state_size'], name='W_smP1')
            self.W_smP2 = self.random_weight(options['state_size'], options['state_size'], name='W_smP2')
            self.W_g_SM = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_g_SM')
            self.W_ruQ = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_ruQ')
            self.W_vQ = self.random_weight(options['state_size'], 2 * options['state_size'], name='W_vQ')
            self.W_VrQ = self.random_weight(options['q_length'], options['state_size'],
                                            name='W_VrQ')  # has same size as u_Q
            self.W_hP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_hP')
            self.W_ha = self.random_weight(2 * options['state_size'], options['state_size'], name='W_ha')
            # fc
            self.W_fc = self.random_weight(2 * options['state_size'], options['emb_dim'], name='W_fc')

            # Biases
            self.B_v_QP = self.random_bias(options['state_size'], name='B_v_QP')
            self.B_v_SM = self.random_bias(options['state_size'], name='B_v_SM')
            self.B_v_rQ = self.random_bias(2 * options['state_size'], name='B_v_rQ')
            self.B_v_ap = self.random_bias(options['state_size'], name='B_v_ap')

            # QP_match
            with tf.variable_scope('QP_match') as scope:
                self.QPmatch_cell = self.DropoutWrappedGRUCell(self.options['state_size'], self.options['in_keep_prob'])
                self.QPmatch_state = self.QPmatch_cell.zero_state(self.options['batch_size'], dtype=tf.float32)

            # Ans Ptr
            with tf.variable_scope('Ans_ptr') as scope:
                self.AnsPtr_cell = self.DropoutWrappedGRUCell(2 * self.options['state_size'],
                                                              self.options['in_keep_prob'])

    def build_model(self):
        opts = self.options

        # placeholders
        paragraph = tf.placeholder(tf.float32, [opts['batch_size'], opts['p_length'], opts['emb_dim']])
        question = tf.placeholder(tf.float32, [opts['batch_size'], opts['q_length'], opts['emb_dim']])
        answer = tf.placeholder(tf.float32, [opts['batch_size'], 3, opts['p_length'], opts['emb_dim']])

        print('process answer to a vector')
        a_embedding = tf.reshape(answer, [opts['batch_size'] * 3, opts['p_length'], opts['emb_dim']])
        a_attention = self.mat_weight_mul(a_embedding, self.W_ans2_1)
        a_score = tf.nn.softmax(a_attention, axis=1)
        a_output = tf.squeeze(tf.matmul(tf.transpose(a_score, [0, 2, 1]), a_embedding))
        a_embedding = tf.reshape(a_output, [opts['batch_size'], 3, opts['emb_dim']])

        print('Question and Passage Encoding')
        eQcQ = question
        ePcP = paragraph

        unstacked_eQcQ = tf.unstack(eQcQ, opts['q_length'],
                                    1)  # [q_length, batch_size, emb_dim],其中q_length就是RNN的time_step
        unstacked_ePcP = tf.unstack(ePcP, opts['p_length'],
                                    1)  # [p_length, batch_size, emb_dim],p_length就是RNN的time_step
        with tf.variable_scope('encoding') as scope:
            stacked_enc_fw_cells = [self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in
                                    range(2)]
            stacked_enc_bw_cells = [self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in
                                    range(2)]
            q_enc_outputs, q_enc_final_fw, q_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_eQcQ, dtype=tf.float32, scope='context_encoding')
            tf.get_variable_scope().reuse_variables()
            p_enc_outputs, p_enc_final_fw, p_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_ePcP, dtype=tf.float32, scope='context_encoding')
            u_Q = tf.stack(q_enc_outputs, 1)  # 回到[batch_size, q_length, 2*opts['state_size']], q_length就是RNN的time_step
            u_P = tf.stack(p_enc_outputs, 1)  # 同理
        u_Q = tf.nn.dropout(u_Q, opts['in_keep_prob'])
        u_P = tf.nn.dropout(u_P, opts['in_keep_prob'])
        print(u_Q)
        print(u_P)

        v_P = []
        print('Question-Passage Matching')
        for t in range(opts['p_length']):
            # Calculate c_t
            W_uQ_u_Q = self.mat_weight_mul(u_Q, self.W_uQ)  # [batch_size, q_length, opts['state_size']]
            tiled_u_tP = tf.concat([tf.reshape(u_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
            W_uP_u_tP = self.mat_weight_mul(tiled_u_tP, self.W_uP)

            if t == 0:
                # 当t==0时，初始状态，不存在V_t-1_P
                tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)  # [batch_size, q_length, opts['state_size']]
            else:
                tiled_v_t1P = tf.concat([tf.reshape(v_P[t - 1], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
                W_vP_v_t1P = self.mat_weight_mul(tiled_v_t1P, self.W_vP)
                tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)  # [batch_size, q_length, opts['state_size']]

            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_QP, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'],
                                  2)  # [batch_size, q_length, 2*opts['state_size']]
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1)  # [batch_size, 2 * state_size]

            # gate
            u_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(u_P[:, t, :]), c_t], 1),
                                      1)  # [batch_size, 1, 4*opts['state_size']]
            g_t = tf.sigmoid(self.mat_weight_mul(u_tP_c_t, self.W_g_QP))
            u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t))

            # QP_match
            with tf.variable_scope("QP_match"):
                if t > 0: tf.get_variable_scope().reuse_variables()
                output, self.QPmatch_state = self.QPmatch_cell(u_tP_c_t_star, self.QPmatch_state)
                v_P.append(output)
        v_P = tf.stack(v_P, 1)  # [batch_size, p_length, opts['state_size']]
        v_P = tf.nn.dropout(v_P, opts['in_keep_prob'])
        print('v_P', v_P)

        print('Self-Matching Attention')
        SM_star = []
        for t in range(opts['p_length']):
            # Calculate s_t
            W_p1_v_P = self.mat_weight_mul(v_P, self.W_smP1)  # [batch_size, p_length, state_size]
            tiled_v_tP = tf.concat([tf.reshape(v_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
            W_p2_v_tP = self.mat_weight_mul(tiled_v_tP, self.W_smP2)

            tanh = tf.tanh(W_p1_v_P + W_p2_v_tP)
            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_SM, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * opts['state_size'], 2)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, v_P), 1)  # [batch_size, 2 * state_size]

            # gate
            v_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(v_P[:, t, :]), c_t], 1), 1)
            g_t = tf.sigmoid(self.mat_weight_mul(v_tP_c_t, self.W_g_SM))
            v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))
            SM_star.append(v_tP_c_t_star)
        SM_star = tf.stack(SM_star, 1)
        unstacked_SM_star = tf.unstack(SM_star, opts['p_length'], 1)
        with tf.variable_scope('Self_match') as scope:
            SM_fw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
            SM_bw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
            SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell,
                                                                                           unstacked_SM_star,
                                                                                           dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1)  # [batch_size, p_length, 2 * opts['state_size']]
        h_P = tf.nn.dropout(h_P, opts['in_keep_prob'])
        print('h_P', h_P)

        print('Output Layer')
        # calculate r_Q  (初始化 h_-1_a)
        W_ruQ_u_Q = self.mat_weight_mul(u_Q, self.W_ruQ)  # [batch_size, q_length, 2 * state_size]
        W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_vQ)
        W_vQ_V_rQ = tf.stack([W_vQ_V_rQ] * opts['batch_size'], 0)  # stack -> [batch_size, q_length, 2 * state_size]

        tanh = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)
        s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_rQ, [-1, 1])))
        a_t = tf.nn.softmax(s_t, 1)
        tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'], 2)
        r_Q = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1)  # [batch_size, 2 * state_size]
        r_Q = tf.nn.dropout(r_Q, opts['in_keep_prob'])
        print('r_Q', r_Q)

        # predition
        W_hP_h_P = self.mat_weight_mul(h_P, self.W_hP)  # [batch_size, p_length, state_size]
        h_t1a = r_Q
        print('h_t1a', h_t1a)
        tiled_h_t1a = tf.concat([tf.reshape(h_t1a, [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
        W_ha_h_t1a = self.mat_weight_mul(tiled_h_t1a, self.W_ha)

        tanh = tf.tanh(W_hP_h_P + W_ha_h_t1a)
        s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_ap, [-1, 1])))
        a_t = tf.nn.softmax(s_t, 1)
        tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'], 2)
        c_t = tf.reduce_sum(tf.multiply(tiled_a_t, h_P), 1)  # [batch_size, 2 * state_size]
        result = tf.expand_dims(c_t, axis=1)
        pred_vec = tf.nn.leaky_relu(self.mat_weight_mul(result, self.W_fc))  # [batch_size, 1, emb_dim]
        pred_vec = tf.nn.dropout(pred_vec, opts['in_keep_prob'])
        final_output = tf.squeeze(
            tf.matmul(a_embedding, tf.transpose(pred_vec, [0, 2, 1])))  # [batch_size, 1, 3] --> [batch_size, 3]
        score = tf.nn.softmax(final_output, axis=1)
        loss = -tf.reduce_sum(tf.log(score[:, 0] + 0.0000001))

        input_tensors = {
            'p': paragraph,
            'q': question,
            'a': answer,
        }

        print('Model built')
        for v in tf.global_variables():
            print(v.name, v.shape)

        return input_tensors, loss, final_output

