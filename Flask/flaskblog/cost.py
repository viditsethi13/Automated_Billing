cost=100;fruit = input("Enter Fruit name:") 
print(fruit)
if fruit == 'apricot':
    cost=100;

elif fruit == 'beetroot':
    cost=10;
elif fruit ==  'blueberry':
    cost=12;

elif fruit == 'cauliflower':
    cost=1;
    
elif fruit == 'dates':
    cost=15;
    
elif fruit ==  'ginger_root':
    cost=100;             
    
elif fruit ==  'guava':
    cost=100;
    
elif fruit ==  'kiwi':
    cost=100;   
    
elif fruit =='lychee':
    cost=100;   
    
elif fruit == 'orange':
    cost=100;
    
elif fruit =='papaya':
    cost=100;
    
elif fruit == 'rasbery':
    cost=100;
    
elif fruit =='walnut':
    cost=100;
    
elif fruit =='apple':
    cost=100;
    
elif fruit =='pineapple':
    cost=100;

elif fruit =='strawberry':
    cost=100;          

print(cost)
quantity=input("Enter Quantity:")
print(quantity)
quantity= int(quantity)
cost= int(cost)
final_cost= quantity * cost
print(final_cost)