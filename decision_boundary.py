# create classifier with different kernels
from sklearn import svm

clf_linear = svm.SVC(kernel="linear")
clf_rbf = svm.SVC(kernel="rbf")
clf_poly3 = svm.SVC(kernel="poly", degree=3)
clf_poly10 = svm.SVC(kernel="poly", degree=10)

titles = ("Linear", "RBF", "Polynomial, degree = 3", "Plynomial, degree = 10")

# create a mesh to plot in
xx, yy = np.meshgrid(np.arange(-3, 3, 0.01), np.arange(-3, 3, 0.01))

# loop over classifiers
for i, clf in enumerate((clf_linear, clf_rbf, clf_poly3, clf_poly10)):
  # train classifier - assign -1 label for C01 and 1 for C02
  clf.fit(np.concatenate((C01, C02), axis=0), [-1]*len(C01) + [1]*len(C02))

  # visualize results
  plt.subplot(2, 2, i + 1,)

  # decision boundary
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap='Blues')

  # training data
  plt.scatter(*C01.T, marker='.')
  plt.scatter(*C02.T, marker='.')

  plt.title(titles[i])

plt.tight_layout()