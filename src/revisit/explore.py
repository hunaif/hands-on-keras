import matplotlib.pyplot as plt
def explore(data):
    sex_pivot = data.pivot_table(index='Sex',values='Survived')
    sex_pivot.plot.bar()
    plt.show()

    pclass_pivot = data.pivot_table(index='Pclass',values='Survived')
    pclass_pivot.plot.bar()
    plt.show()


# age_pivot = train.pivot_table(index='Age_categories',values='Survived',aggfunc='mean')
# age_pivot.plot.bar()
# plt.show()
# print(train['Pclass'].value_counts())
