# 🛡️ CNN Adversarial Attack Research
### *Exploring the Fragility of Deep Neural Networks through FGSM*

---

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

**From 98.8% Accuracy To 9.03% Accuracy with ε=0.3**

</div>

---

## 📊 **Project Overview**

This repository demonstrates the **vulnerability of Convolutional Neural Networks** to adversarial attacks using the **Fast Gradient Sign Method (FGSM)**. What starts as a high-performing MNIST classifier quickly crumbles under carefully crafted perturbations.

> *"The ease with which we can switch between these two worlds is remarkable. In one world, the neural network is confident and accurate. In the other, the same network is confused and wrong."* - Ian Goodfellow

---

## 🏗️ **Model Architecture**

Our CNN follows a **simple yet effective** design philosophy:

```
📥 Input Layer (28×28×1)
    ↓
🔍 Conv2D(1→8, 3×3) + ReLU
    ↓
📉 MaxPool2D(2×2)
    ↓
🔍 Conv2D(8→16, 3×3) + ReLU
    ↓
📉 MaxPool2D(2×2)
    ↓
🧠 Fully Connected(16×7×7 → 10)
    ↓
📤 Output (10 classes)
```

### **Implementation Details**
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Dataset**: MNIST (28×28 grayscale images)
- **Training Accuracy**: **98.8%** ✨

---

## ⚔️ **Adversarial Attack: FGSM**

### **What is FGSM?**

The **Fast Gradient Sign Method** is a white-box adversarial attack that generates perturbations by:

1. Computing the gradient of the loss w.r.t. input image
2. Taking the sign of the gradient
3. Scaling by epsilon (ε) to control perturbation magnitude

**Mathematical Formula:**
```
x' = x + ε × sign(∇ₓ J(θ, x, y))
```

Where:
- `x'` = adversarial example
- `x` = original input
- `ε` = perturbation magnitude
- `J` = loss function

### **📚 Research Paper**
🔗 [**Explaining and Harnessing Adversarial Examples**](https://arxiv.org/abs/1412.6572) - Goodfellow et al., 2014

---
## 📷 **Perturbed Images at Increasing Epsilon**

Even minimal perturbations can significantly alter model predictions while remaining nearly imperceptible to the human eye.

<div align="center">

<img src="https://raw.githubusercontent.com/Achintya47/Achintya47/main/perturbed_images.png" alt="Adversarial Examples at Varying Epsilon" width="80%"/>

</div>

> 🧠 This visualization clearly illustrates how increasing epsilon (ε) gradually distorts the image, confusing the model.

## 🔥 **Shocking Results**


The model's performance **dramatically degrades** as epsilon increases:

| Epsilon (ε) | Test Accuracy | Accuracy Drop | Status |
|-------------|---------------|---------------|--------|
| **0.00** | **98.74%** | - | 🟢 Pristine |
| **0.05** | **93.03%** | ↓ 5.71% | 🟡 Slight Impact |
| **0.10** | **73.87%** | ↓ 24.87% | 🟠 Moderate Impact |
| **0.15** | **45.32%** | ↓ 53.42% | 🔴 Severe Impact |
| **0.20** | **25.22%** | ↓ 73.52% | 🔴 Critical |
| **0.25** | **14.44%** | ↓ 84.30% | 🔴 Near Failure |
| **0.30** | **9.03%** | ↓ 89.71% | 🔴 Complete Failure |

### **📈 Visual Impact**

```
Accuracy vs Epsilon
100% ████████████████████████████████ 98.74% (ε=0.00)
 90% ██████████████████████████████   93.03% (ε=0.05)
 80% ████████████████████████████     73.87% (ε=0.10)
 70% ███████████████████████          45.32% (ε=0.15)
 60% ██████████████████               25.22% (ε=0.20)
 50% █████████████                    14.44% (ε=0.25)
 40% ████████████                     9.03%  (ε=0.30)
 30% ███████████
 20% ████████
 10% ████
  0% ██
```

---

## 🚀 **Key Insights**

> **🔍 Critical Finding**: Even with **ε = 0.1** (barely perceptible to human eye), the model accuracy drops from **98.8%** to **73.9%** - a catastrophic **24.9%** decrease!

### **Why This Matters**
- **Security Implications**: Real-world AI systems are vulnerable
- **Robustness**: High accuracy ≠ robust model
- **Adversarial Training**: Need for defensive mechanisms

---

## 🔧 **Getting Started**

### **Prerequisites**
```bash
pip install torch torchvision numpy matplotlib
```

### **Quick Run**
```bash
git clone https://github.com/Achintya47/Fast-Gradient-Sign-Attack
cd Fast-Gradient-Sign-Attack
jupyter notebook Training_and_Attacking.ipynb
```

---

## 📚 **Research Journey**

This project is part of my **summer research exploration** implementing cutting-edge papers in PyTorch. 

### **🔗 Other Implementations**
- **[Actor-Critic Algorithm from Scratch](https://github.com/Achintya47/Reinforcement-Learning/tree/main/Actor_Critic_Policy_from_scratch)** - Reinforcement Learning implementation
- *More projects coming soon...*

---

## 🤝 **Connect & Collaborate**

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in\achintyasharma47)

**Let's discuss AI security and robustness!**

</div>

---

## 📖 **References**

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv preprint arXiv:1412.6572.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE.

---

<div align="center">

### *"In adversarial examples, we see the gap between human and machine perception "*

**⭐ Star this repo if you found it insightful!**

</div>

---

