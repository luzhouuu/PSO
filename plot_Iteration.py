import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
df = pd.read_csv("Iteration.csv")

funcnamelist = ["X-Squared", "Booth", "Beale", "ThreeHumpCamel", "GoldsteinPrice",
                "Levi_n13", "Sphere", "Rosebrock", "StyblinskiTang", "Ackley", "Schaffer_n2", 
                "Eggholder", "McCormick", "Rastrigin", "Schaffer_n4", "Easom", "Bukin_n6", "Matyas"]

set(df["Function"])
for f in funcnamelist:
    data = df[df["Function"] == f]
    plt.figure(figsize=(10,5))
    plt.plot(data[data["Method"]=="PSO"]["times"],data[data["Method"]=="PSO"]["error"])
    plt.plot(data[data["Method"]=="QPSO"]["times"],data[data["Method"]=="QPSO"]["error"])
    plt.ylabel("Error")
    plt.xlabel("Number of Iteration")
    plt.title("Plot (gbest - true min) for" + " "+  str(f))
    plt.legend(["PSO", "QPSO"]);