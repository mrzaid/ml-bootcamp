import numpy as np

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000
    n=len(x)
    learning_rate=0.001
    for i in range(iterations):
        y_pred=m_curr*x + b_curr
        md=-(2/n)*sum(x*(y-y_pred))
        bd=-(2/n)*sum((y-y_pred))
        m_curr=m_curr- learning_rate* md
        b_curr=b_curr- learning_rate* bd