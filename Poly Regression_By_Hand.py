# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:35:46 2018

@author: Luke Hardcastle
"""
import numpy as np  # Here i have imported 3 libraries one is numpy which is essential to this code to do the maths and there is pandas library that i used to pull the csv file 
import pandas
import matplotlib.pyplot as plt # Also matplotlib i used to plot and show my graphs 



a = pandas.read_csv('\\UNI\\Year 2\\Algorithms for data mining\\regression_train_assignment2017.csv') ## Important! To get the csv file to read you need to change the path "UNI\\Year 2\\Algorithms for data mining" to the location of where you have saved the file regression_train_assignment2017.csv for example, C:\\Users\\USER_NAME\\Downloads\\regression_train_assignment2017.csv
a = np.asarray(a)

x = a[:,1] # Here picks the array values which is left blank as that means it uses the full table, this will represent data in the x column
y = a[:,2] # this will represent data in the y column

def poly_regression(x_train, y_train, degree): #This is my function i have created to do poly regression
 
 #xmatrix = np.asmatrix(x) #converts X values in to a numpy matrix this is called xmatrix
 
 ###Here is the values that are used to work out what the power of 2,3 ect would be on the data set x, the input must be a square array so therefor i needed to use x and square and cube ect, those values.
 square2 = pow(x,2)
 cubed= pow(x,3)
 four = pow(x,4)
 five = pow(x,5)
 six = pow(x,6)
 seven = pow(x,7)
 eight = pow(x,8)
 nine = pow(x,9)
 ten = pow(x,10)
 ### Below i have writen code to Regress a polynomial of the following degrees: 0, 1, 2, 3, 5, 10 so to make it easier to debug i did 0-10.
 ### Here i have taken the, squared, cubed ect values, from earlier on and turned them back in to a Matrix "expression" so this means it is simlar to a matix format, i have done this by using np.vstack which can stacks arrays and data on top of one anoter creating a 1-D arrays
 degreezero = np.vstack([np.ones(len(x))]) #Here i have defined the deggree of 0 to get the correct results for this polynomial function as it is required 
 degreeone = np.vstack([np.ones(len(x)),x]) #This code use's, x data, it uses vstack so each line is above one another and then it assign's 1. values to the each bit of data using np.ones(len(x)) then (len(x)) tells it to do it on the lenght of x this has been done as its needed for the matrix equation
 degreetwo = np.vstack([degreeone,square2])
 degreethree = np.vstack([degreetwo,cubed])# Then for the rest of the code it is the creation of the future degree values, I have used degreeone as the base value as it holds x's data, that has been stacked and np.oneed earlier on in the code 
 degreefour = np.vstack([degreethree,four]) ## then i make use of the np.vstack and have for example, the four value which has been made ealier on where i took x's data and got an output to the power of 4 and stacked it against degreeone
 degreefive = np.vstack([degreefour,five])
 degreesix = np.vstack([degreefive,six])
 degreeseven = np.vstack([degreesix,seven])
 degreeeight = np.vstack([degreeseven,eight])
 degreenine = np.vstack([degreeeight,nine])
 degreeten = np.vstack([degreenine,ten])
 ### Here shows I have taken the stacked degrees from above and used numpy's function, asmatrix to turn the stacked arrarys in to matrix's, then i have trasposed the matrix so it orders the data in to coloum line by line.
 degree0 =np.asmatrix(degreezero).T
 degree1 =np.asmatrix(degreeone).T
 degree2 =np.asmatrix(degreetwo).T
 degree3 =np.asmatrix(degreethree).T
 degree4 =np.asmatrix(degreefour).T
 degree5 =np.asmatrix(degreefive).T
 degree6 =np.asmatrix(degreesix).T
 degree7 =np.asmatrix(degreeseven).T
 degree8 =np.asmatrix(degreeeight).T
 degree9 =np.asmatrix(degreenine).T
 degree10 =np.asmatrix(degreeten).T
 y=np.asmatrix(y_train).T # Here the code is taking y_train and turning it in to a matrix then transposing it and assigning that code to y
 ### Here I used the lecture slides to figure out the least square estimation which is explained in the assingment document, but what this code is doing is taking degree1 and transposing it then timesing it by degree1 and then inverts the output, then times by degree1 which is transposed then degree1 is times by y which was created just above.
 y=np.asmatrix(y_train).T
 d0 = (degree0.T*degree0).I*degree0.T*y
 d1 = (degree1.T*degree1).I*degree1.T*y
 d2 = (degree2.T*degree2).I*degree2.T*y
 d3 = (degree3.T*degree3).I*degree3.T*y
 d4 = (degree4.T*degree4).I*degree4.T*y
 d5 = (degree5.T*degree5).I*degree5.T*y
 d6 = (degree6.T*degree6).I*degree6.T*y
 d7 = (degree7.T*degree7).I*degree7.T*y
 d8 = (degree8.T*degree8).I*degree8.T*y
 d9 = (degree9.T*degree9).I*degree9.T*y
 d10 = (degree10.T*degree10).I*degree10.T*y
 
 ### Here the code is making use of the degree parameter that is included above in the creation of the function, poly_regression, i used if statments to define each degree from 1-10 so for example, from 1-10 what ever number that is put in poly_regression(x,y,0) at the 0 point it will return the related value that you have entered, these are the parameters
 if degree == 0:
     return(d0) #This line of code then returns parmater d0 if the degree input in the call of this function equals 0
 if degree == 1:
     return(d1)
 if degree == 2:
     return(d2) 
 if degree == 3:
     return(d3)
 if degree == 4:
     return(d4)
 if degree == 5:
     return(d5)  
 if degree == 6:
     return(d6)  
 if degree == 7:
     return(d7)
 if degree == 8:
     return(d8)
 if degree == 9:
     return(d9)
 if degree == 10:
     return(d10)



def eval_poly_regression(poly,x,y,degree): ## here is where the the new function for evaluation algorithm is made.
    y = np.matrix(y).T #Here is just taking y's data and puting it in to a matrix and then transposing it
    degreezero = np.vstack([np.ones(len(x))])
    degreeone = np.vstack([np.ones(len(x)),x])
    degreetwo = np.vstack([degreeone,pow(x,2)])# This is pretty much the same code from above apart from e.g. the code takes degreeone and uses it to do x to the power of 2 which gives us the next value to be able to use in the sequence.
    degreethree = np.vstack([degreetwo,pow(x,3)])
    degreefour = np.vstack([degreethree,pow(x,4)]) 
    degreefive = np.vstack([degreefour,pow(x,5)])
    degreesix = np.vstack([degreefive,pow(x,6)])
    degreeseven = np.vstack([degreesix,pow(x,7)])
    degreeeight = np.vstack([degreeseven,pow(x,8)])
    degreenine = np.vstack([degreeeight,pow(x,9)])
    degreeten = np.vstack([degreenine,pow(x,10)])
    ### This code herte is an if statement to take the values from above and do some maths on them to find out what the root meen square error is, these if statements range between 0-10
    if degree == 0:
     a=degreezero.T*poly
     predic0 = a-y
     predic0=np.square(predic0) #predic0 is just a variable and is not really significant to change, it just stores the calculations
     predic0=np.mean(predic0)
     rmse=np.sqrt(predic0)
     return(rmse)
    if degree == 1:
     a=degreeone.T*poly # a=degreezero.T*poly this is all that needs to be changed for each if statement you just change what degree to use relevent to the degree you want to show
     predic0 = a-y # This takes A and minus's it away from y
     predic0=np.square(predic0) # This uses numpy and squares the value predic0
     predic0=np.mean(predic0) # This line in the if statements works out the mean of predic0 value
     predic0=np.sqrt(predic0) # Then this line works out the sqrt of predic0 using the predic0 from the line above
     return(predic0) #This line then returns predic0 if the degree equals 1 in the call of this function eval_poly_regression(poly,x,y,degree) this gives us the root mean squared error
    if degree == 2:
     a=degreetwo.T*poly
     predic0 = a-y
     predic0=np.square(predic0)
     predic0=np.mean(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 3:
     a=degreethree.T*poly
     predic0 = a-y
     predic0=np.square(predic0)
     predic0=np.mean(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 4:
     a=degreefour.T*poly
     predic0 = a-y
     predic0=np.square(predic0)
     predic0=np.mean(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 5:
     a=degreefive.T*poly
     predic0 = a-y
     predic0=np.square(predic0)
     predic0=np.mean(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 6:
     a=degreesix.T*poly
     predic0 = a-y
     predic0=np.square(predic0)
     predic0=np.mean(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)  
    if degree == 7:
     a=degreeseven.T*poly
     predic0 = a-y
     predic0=np.mean(predic0)
     predic0=np.square(predic0)
     predic0=np.sqrt(predic0).T
     return(predic0)
    if degree == 8:
     a=degreeeight.T*poly
     predic0 = a-y
     predic0=np.mean(predic0)
     predic0=np.square(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 9:
     a=degreenine.T*poly
     predic0 = a-y
     predic0=np.mean(predic0)
     predic0=np.square(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
    if degree == 10:
     a=degreeten.T*poly
     predic0 = a-y
     predic0=np.mean(predic0)
     predic0=np.square(predic0)
     predic0=np.sqrt(predic0)
     return(predic0)
     
poly = poly_regression(x,y,1) #This is the call function to the function poly_regression(x_train, y_train, degree): these functions need to be here for it to work // change the number at the end of each call to pick what degree you want
eval_poly = eval_poly_regression(poly,x,y,1) #This is the call function to the function eval_poly_regression(poly,x,y,degree) // change the number at the end of each call to pick what degree you want
print(eval_poly)# I have put this here to help show what the outputs are for each function 
print(poly) # I have put this here to help show what the outputs are for each function 

poly=np.asarray(poly)
poly=np.squeeze(poly)
poly=np.flip(poly,0) # Comment out this if you want to find degree 0 

pol = np.poly1d(poly)

plt.scatter(x, y,  color='black')
xp = np.linspace(-5, 5, 100)
plt.plot(xp,pol(xp)) #This code takes the xp value from above and tells the graph what it needs to plot and where as you can see it used xp which is the linspacing and it uses pol which is the out come of np.poly1d
plt.ylabel('Y')
plt.xlabel('X')
plt.grid(True)

plt.show() # This shows the graph thats been created
print(pol) # This print is here to show you the vaules that are beeing used in the graph 
