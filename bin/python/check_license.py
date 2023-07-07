import gurobipy as gp

# create a small model
m = gp.Model()

x = m.addVar()
y = m.addVar()
m.setObjective(x + y, gp.GRB.MAXIMIZE)
m.addConstr(x + 2 * y <= 1)

# solve the model and print the log
m.optimize()
