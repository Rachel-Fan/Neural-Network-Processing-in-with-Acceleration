#include "helpers.h"
#include <iostream>

void printResult(const std::vector<double> &result)
{
    for (auto &value : result)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}
