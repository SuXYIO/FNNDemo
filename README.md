# FNNDemo
## Note
**The versions with this note is not completed yet, still in development.**  
Adopted from SuXYIO/LRDemo.  

## Intro
Simple demo to FNN using C.  

## Install
### Linux/Unix:
Clone:
```bash
git clone https://github.com/SuXYIO/FNNDemo
```
Compile:
```
cd FNNDemo
make
```
### Windows:
Not supported yet.  
The program uses POSIX libs, which vanilla Windows doesn't support.  
Might work if using methods like MinGW to use POSIX standards on Windows.  

## Usage
```bash
./FNN.out [options]
```
### Args
See manual:
```bash
./FNN.out -h
```
### Other
None, yet.  

## Technical Details
### Algorithms
Loss function: `MSE`  
Regression algorithm: `Gradient decent`  
Random algorithm: `Box-muller normal distro`  

## Purpose
A project to help improve C programming skills & experiment with and understand neural networks better.  

