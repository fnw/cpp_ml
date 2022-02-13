# cpp_ml
A repository where I keep the Machine Learning algorithms I've written in C++. This is mostly based on the algorithms I've written previously in Python at [this repo](https://github.com/fnw/python_ml).

One of the goals of starting this little exploration was getting to work with the [Eigen](https://eigen.tuxfamily.org/) library. I'm still wrapping my head around it, so this might not be the prettiest Eigen code you've ever seen.

## Running the code

The code assumes that Eigen is available on `./Eigen`.
To compile the code, use `g++ -I ./Eigen file.cpp` and then run `a.out`. You might also want to turn on optimizations, as per the recommendations of the Eigen developers.

If you have Eigen installed system-wide, just change the includes and remove the -I flag from the compiler invocation.


## License
BSD 2-clause.