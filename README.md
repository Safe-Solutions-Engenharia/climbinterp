# ClimbInterp

This interpolation method aims to maintain conservatism across a set of data.
The strategy used is to assume exponential behaviour from a certain point, however until the interpolation gets to this point,
other forms of interpolation are used and interconnected, creating two functions that complement each other.

## Instalation

Use the package manager [pip](https://pip.pypa.io/en/stable) to install climbinterp.

```bash
pip install climbinterp
```

## Usage

```python
from climbinterp.data_arrange import ArrangeData
from climbinterp.adv_interpolation import ClimbInterp
from random import sample

# Random set of x and y data.
random_numbers_x = sample(range(1, 51), 50)
random_numbers_y = sample(range(1, 51), 50)

# Will arrange the data in ascending order.
arrange_data = ArrangeData(random_numbers_x, random_numbers_y)
random_numbers_x, random_numbers_y = arrange_data.get_arranged_points()

climb_interp = ClimbInterp(random_numbers_x, random_numbers_y)
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
