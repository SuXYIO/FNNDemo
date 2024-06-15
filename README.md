# LR
## Note
**The versions with this note is not completed yet, still in development.**  
Adopted from SuXYIO/LR.  

## Intro
Simple demo to MLP using C.  

## Install
### Linux/Unix:
Clone:
```bash
git clone https://github.com/SuXYIO/MLPDemo
```
Compile:
```
cd MLPDemo
make
```
### Windows:
Not supported yet.  
The program uses POSIX libs, which vanilla Windows doesn't support.  
Might work if using methods like MinGW to use POSIX standards on Windows.  

## Usage
```bash
./MLP.out [options]
```
### Args
See manual:
```bash
./MLP.out -h
```
### Other
None, yet.  

## Technical Details
### Algorithms
Loss function: `MSE`  
Regression algorithm: `Gradient decent`  
Random algorithm: `Box-muller normal distro`  

## Purpose
A project to help improve C programming skills & experiment with and understand MLPs better.  

