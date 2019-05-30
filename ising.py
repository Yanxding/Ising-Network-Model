"""Author: Yanxiang Ding
Date: 5/30/19
Name: ising.py
Version: 1.0
Function: This module is used for learning an Ising model.
Documentation: https://github.com/Yanxding/Ising-Network-Model
Contact: yanxiangding914@gmail.com
"""

class ising_model:
    
    '''
    initialisation of the class
    '''
    def __init__(self, penalty='l1', plot=True, l1_ratio=1, gamma=0.25, model_selection='local'):
        '''
        penalty: penalty type in model learning, 'l1', 'l2' or 'elasticnet', default = 'l1'
        plot: logic variable, either True of False, determine whether to plot the learned network, default = True
        l1_ratio: trade-off parameter only used for elasticnet penalty, default = None
        gamma: BIC parameter, default = 0.25
        model_selection: 'local' or 'global', whether regularization strength is optimized for each node or for all nodes
        '''
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.plot = plot
        self.model_selection = model_selection
    
    
    '''
    fit the model according to the given training data and specified parameters
    '''
    def fit(self, X):
        
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        import networkx as nx
        import math
        
        def BIC(coef_, intercept_, X, i, gamma):
            EBIC = -2*log_likelihood(coef_, intercept_, X, i) + np.count_nonzero(coef_)*math.log(np.size(X,0)) + 2*gamma*np.count_nonzero(coef_)*math.log(np.size(X,1)^2-1)
            return EBIC
            
        def log_likelihood(coef_, intercept_, X, i):
            a = np.multiply( (np.matmul(X, coef_) + intercept_), X[:,i] )
            b = np.array(list(map(math.exp, (np.matmul(X, coef_) + intercept_)*1))) + np.array(list(map(math.exp, (np.matmul(X, coef_) + intercept_)*(-1))))
            return sum(a - b)

        def net_plot(self):
            adjcency = (self.Omega + np.transpose(self.Omega))/2
            G = nx.from_numpy_matrix(adjcency)
            pos = nx.random_layout(G)
            nx.draw_networkx_nodes(G.nodes(), pos, node_color='green', alpha=0.7)
            edge_list = list(G.edges())
            edge_weight = [G.edges[edge]['weight']*20 for edge in edge_list]
            edge_color = ['r' if G.edges[edge]['weight']<0 else 'b' for edge in edge_list]
            for i in range(len(edge_list)):
                nx.draw_networkx_edges(G, pos, edgelist=[edge_list[i]], edge_color=edge_color[i], width=edge_weight[i])
            nx.draw_networkx_labels(G, pos, font_color='white')
        
        n_features = np.size(X,1)
        self.n = n_features
        Tau = []
        Omega = []
        C = []
        if self.model_selection == 'local':
            for i in range(n_features):
                y = X[:,i]
                X_new = np.delete(X, i, axis=1)
                tau_i = []
                omega_i = []
                EBIC_i = []
                C_val = [0.5,1,5,10,100]
                for c in C_val:
                    logistic = LogisticRegression(penalty=self.penalty, C=c, max_iter=500, solver='saga', l1_ratio=self.l1_ratio)
                    logistic.fit(X_new, y)
                    tau_i.append(logistic.intercept_.tolist()[0])
                    coef_i = logistic.coef_.tolist()[0]
                    coef_i = coef_i[0:i] + [0] + coef_i[i:]
                    omega_i.append(coef_i)
                    EBIC_i.append(BIC(coef_i, logistic.intercept_, X, i, self.gamma))
                best = EBIC_i.index(min(EBIC_i))
                Tau.append(tau_i[best])
                Omega.append(omega_i[best])
                C.append(C_val[best])
        elif self.model_selection == 'global':
            C_val = [0.5,1,5,10,100]
            tau = []
            omega = []
            EBIC_c = []
            for c in C_val:
                tau_c = []
                omega_c = []
                EBIC_i = 0
                for i in range(n_features):
                    y = X[:,i]
                    X_new = np.delete(X, i, axis=1)
                    logistic = LogisticRegression(penalty=self.penalty, C=c, max_iter=500, solver='saga', l1_ratio=self.l1_ratio)
                    logistic.fit(X_new, y)
                    tau_c.append(logistic.intercept_.tolist()[0])
                    coef_c = logistic.coef_.tolist()[0]
                    coef_c = coef_c[0:i] + [0] + coef_c[i:]
                    omega_c.append(coef_c)
                    EBIC_i += BIC(coef_c, logistic.intercept_, X, i, self.gamma)
                tau.append(tau_c)
                omega.append(omega_c)
                EBIC_c.append(EBIC_i)
            best = EBIC_c.index(min(EBIC_c))
            Tau = tau[best]
            Omega = omega[best]
            C = C_val[best]
            
        self.Tau = Tau
        self.Omega = Omega
        self.C = C
        if self.plot == True:
            net_plot(self)
    
    
    '''
    return the approximate total log-likelihood of the given data
    '''
    def score(self, X):
        import math
        def log_likelihood(coef_, intercept_, X, i):
            a = np.multiply( (np.matmul(X, coef_) + intercept_), X[:,i] )
            b = np.array(list(map(math.exp, (np.matmul(X, coef_) + intercept_)*1))) + np.array(list(map(math.exp, (np.matmul(X, coef_) + intercept_)*(-1))))
            return sum(a - b)
        
        log_likelihood_prox = 0
        for i in range(np.size(X,1)):
            log_likelihood_prox += log_likelihood(self.Omega[i], self.Tau[i], X, i)
        return log_likelihood_prox
    
    
    '''
    compute potential of the given instances according to the learned model
    '''
    def potential(self, X):
        import numpy as np
        import math
        a = np.matmul(X, self.Tau)
        b = np.matmul(np.matmul(X, np.transpose(self.Omega)), np.transpose(X))
        if X.ndim > 1:
            potential = np.array(list(map( math.exp, (a + np.diagonal(b)))))
        else:
            potential = math.exp(a + b)
        return potential
    
    
    '''
    predict the most likely state of node i given information of all other nodes
    '''
    def predict(self, X, i):
        import numpy as np
        import math
        
        def potential(self, X):
            a = np.matmul(X, self.Tau)
            b = np.matmul(np.matmul(X, np.transpose(self.Omega)), np.transpose(X))
            if X.ndim > 1:
                potential = np.array(list(map( math.exp, (a + np.diagonal(b)))))
            else:
                potential = math.exp(a + b)
            return potential
        
        X_plus = np.insert(X, (i-1), 1, axis=1)
        X_minus = np.insert(X, (i-1), -1, axis=1)
        potential_plus = potential(self, X_plus)
        potential_minus = potential(self, X_plus)
        if X.ndim == 1:
            pred = 1 if potential_plus >= potential_minus else -1
        else:
            pred = np.array([1 if potential_plus[i] >= potential_minus[i] else -1 for i in range(len(X))])
        return pred
