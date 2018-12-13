
# coding: utf-8





def classify(keyword2int):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.misc
    from PIL import Image

    import pprint
    pp = pprint.PrettyPrinter(indent = 4)


    # In[18]:


    import sklearn
    from sklearn import datasets
    import skimage.io


    # ## Datasets preparation

    # In[19]:


    def png2vec(filename):
        img = Image.open(filename).convert('L')
        arr = np.array(img)
        return arr


    # In[20]:


    filesetNames = ["%01d" %x for x in range(0,2)]

    # In[21]:

    import os
    images = []
    tgt = []
    count = 0
    img_test = Image.open("./img/0/1.jpg")

    for curFileset in filesetNames:
        curPath = "./img/"+ curFileset + "/"
        for file in os.listdir(curPath):
            curImageVector = png2vec(curPath + file)
            images.append(curImageVector)
            tgt.append(curFileset)
            count += 1
    #end
    print (len(images))

    # In[22]:



    # In[24]:


    from sklearn.model_selection import train_test_split
    from sklearn import model_selection, metrics
    images_np = np.array(images)
    img = images_np.reshape(images_np.shape[0],-1)

    xM, xT, yM, yT = train_test_split(img, tgt, test_size = 0.01)
    print(yT)
    xT = list(xT)

    #map(list,xT)
    xT.extend(xM[0:20])
    yT.extend(yM[0:20])
    print(yT)
    xT = np.array(xT)


    print(yT)

    # # Naive Bayes

    # In[25]:
    # 要传参！！！ from search keyword2int
    # 要传参！！！ from search keyword2int
    #keyword2int = {'dog':0,'cat':1}

    from sklearn import metrics
    from sklearn import naive_bayes
    cls = naive_bayes.GaussianNB()
    cls.fit(xM, yM)
    res = cls.predict(xT)
    print(type(xT[0:1]))
    print(len(xT),len(xT[0]),'result',type(xT))
    print(res)
    print(metrics.confusion_matrix(yT, res),metrics.accuracy_score(yT, res),"GaussianNB")
    #plt.title('这个图片是'+str(list(keyword2int.keys())[0])+'预测结果是'+ str(list(keyword2int.keys())[int(res_test[0])]))
    #plt.imshow(img_test)
    #plt.show()
    # In[26]:


    cls = naive_bayes.BernoulliNB(binarize = 0.9)
    cls.fit(xM, yM)
    res = cls.predict(xT)
    print(metrics.confusion_matrix(yT, res),metrics.accuracy_score(yT, res),"BernoulliNB")


    # In[27]:


    cls = naive_bayes.MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
    cls.fit(xM, yM)
    res = cls.predict(xT)
    print(metrics.confusion_matrix(yT, res),metrics.accuracy_score(yT, res),"naive_bayes.MultinomialNB")


    # # KNN

    # In[28]:


    from sklearn.neighbors import KNeighborsClassifier 
    cls = KNeighborsClassifier(n_neighbors=1)
    cls.fit(xM, yM)
    res = cls.predict(xT)
    print(metrics.confusion_matrix(yT, res),metrics.accuracy_score(yT, res),' knn')


    # # Decision Tree

    # In[33]:


    from sklearn import tree
    cls = tree.DecisionTreeClassifier()
    cls = cls.fit(xM,yM)
    res = cls.predict(xT)
    print(metrics.confusion_matrix(yT,res), metrics.accuracy_score(yT,res),"Decision Tree")
    import random
    index = random.randint(0, 10)
    x_test = xT[index]
    y_tag = yT[index]
    DT_res = cls.predict([x_test])

    x_test_image_list = list(x_test)
    x_temp = []
    x_image = []
    for j in range(100):
        for i in range(100):
            x_temp.append(x_test[i+j*100])
        
        x_image.append(x_temp)
        x_temp = []

    np.array(x_image)
    plt.title('the image is '+str(list(keyword2int.keys())[int(y_tag)])+' result is '+ str(list(keyword2int.keys())[int(DT_res[0])]))
    print(DT_res)
    print(list(keyword2int.keys())[int(y_tag)],list(keyword2int.keys())[int(DT_res[0])])
    plt.imshow(x_image)
    plt.show()

    import sys
    sys.exit()

    # In[40]:


    from sklearn.tree import DecisionTreeClassifier


    tree = DecisionTreeClassifier(random_state=0, min_samples_split=2)
    tree.fit(xT,yT)

    print(tree.score(xT, yT),tree.score(xM, yM),"DecisionTreeClassifier")


    # In[37]:


    DecisionTreeClassifier()


    # # random forest 

    # In[56]:


    from sklearn.ensemble import RandomForestClassifier


    forest = RandomForestClassifier(n_estimators=150, random_state=0)
    forest.fit(xT, yT)


    forest.score(xT, yT),forest.score(xM, yM)


    # In[44]:


    forest


    # In[74]:


    scoreListUniform = []
    nListUniform = []
    stepCount = 10
    stepDist = 200
    start = 200
    end = 2001
    for curCount in range(start,end,stepDist):
        forest = RandomForestClassifier(n_estimators = curCount, random_state=0)
        forest.fit(xT, yT)
        nListUniform.append(curCount)
        scoreListUniform.append(forest.score(xM, yM))
    scoreListUniform




    # ## SVM

    # In[76]:


    from sklearn import svm
    cls = svm.SVC()
    cls = cls.fit(xM,yM)
    res = cls.predict(xT)
    print(metrics.confusion_matrix(yT,res), metrics.accuracy_score(yT,res), " svm")

    from sklearn import svm
    import matplotlib.pyplot as plt
    import numpy
    n_trials = 3
    train_percentages = range(5,95,5)
    test_accuracies = numpy.zeros(len(train_percentages))

    for (i, tp) in enumerate(train_percentages):
        test_accuracy = numpy.zeros(n_trials)
        for n in range(n_trials):
            xM, xT, yM, yT = train_test_split(digits.data, digits.target,  train_size=tp/100.0)
            cls = svm.LinearSVC().fit(xM, yM)
            res = cls.predict(xT)
            test_accuracy[n] = metrics.accuracy_score(yT, res)
        test_accuracies[i] = test_accuracy.mean()
        print(i, tp, test_accuracies[i])
    print( train_percentages)
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()

    # In[77]:
keyword2int = {'pine tree': 0, 'maple leaves': 1, 'sakura blossom': 2}
classify(keyword2int)
