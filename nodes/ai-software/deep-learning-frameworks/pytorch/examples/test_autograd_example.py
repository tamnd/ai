# The test of the example of the pytorch page: the output printed on the page is the output of this program.
from autograd_example import main


def test_output(capsys):
    y, grad, func_grad = main()
    assert y == 14.0
    assert grad == [2.0, 4.0, 6.0]
    assert func_grad == [2.0, 4.0, 6.0]
    assert capsys.readouterr().out == (
        "f(x) = 14.0\n"
        "grad by backward() = [2.0, 4.0, 6.0]\n"
        "grad by torch.func.grad = [2.0, 4.0, 6.0]\n"
    )
