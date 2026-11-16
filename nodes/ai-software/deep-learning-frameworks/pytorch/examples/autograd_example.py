# A small PyTorch program: eager execution with Python control flow, and reverse-mode
# automatic differentiation through it, first with backward() and then with torch.func.grad.
import torch


def f(x):
    # Ordinary Python control flow runs eagerly, one operation at a time.
    if x.sum() > 0:
        return (x**2).sum()
    return (-x).sum()


def main():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = f(x)
    y.backward()
    print("f(x) =", y.item())
    print("grad by backward() =", x.grad.tolist())

    g = torch.func.grad(f)(torch.tensor([1.0, 2.0, 3.0]))
    print("grad by torch.func.grad =", g.tolist())
    return y.item(), x.grad.tolist(), g.tolist()


if __name__ == "__main__":
    main()
