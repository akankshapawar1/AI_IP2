"""
Please enter the origin city: Boston
Could not find Boston, please try again: Arad
Please enter the destination city: Arad
The same city can't be both origin and destination. Please try again.
Please enter the origin city: Arad
Please enter the destination city: Bucharest
Greedy Best-First Search
Arad → Sibiu → Fagaras → Bucharest
Total Cost: 450
A* Search
Arad → Sibiu → Rimnicu → Pitesti → Bucharest
Total Cost: 418
Hill Climbing Search
Arad → Sibiu → Fagaras → Bucharest
Total Cost: 450
Simulated Annealing Search
Arad → Sibiu → Rimnicu → Craiova → Pitesti → Bucharest → Urziceni → Bucharest → Giurgiu → Bucharest
Total Cost: 955
Would you like to find the best path between other two cities? yes
Would you like to find the best path between other two cities? no
Thank You for Using Our App
"""
from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent, romania_map

cities = ['Arad', 'Bucharest', 'Craiova', 'Drobeta', 'Eforie', 'Fagaras', 'Giurgiu', 'Hirsova', 'Iasi', 'Lugoj',
          'Mehadia', 'Neamt', 'Oradea',
          'Pitesti', 'Rimnicu', 'Sibiu', 'Timisoara', 'Urziceni', 'Vaslui', 'Zerind']


def main():
    agent = SimpleProblemSolvingAgent()

    def get_city(prompt):
        while True:
            city = input(prompt)
            if city in cities:
                return city
            else:
                prompt = "Could not find " + city + ", please try again: "

    print("Here are all the possible Romania cities that can be traveled:")
    print(cities)
    while True:
        origin = get_city("Please enter the origin city: ")
        while True:
            destination = get_city("Please enter the destination city: ")
            if origin == destination:
                print("The same city can't be both origin and destination. Please try again. ")
                origin = get_city("Please enter the origin city: ")
            else:
                # SPSA(origin, destination)
                # print("SPSA will be called now")
                agent.find_path(origin, destination, romania_map.graph_dict, romania_map.locations)
                break
        user_continue_choice = input("Would you like to find the best path between other two cities? ")
        if user_continue_choice.lower() == "no":
            print("Thank You for Using Our App")
            break


if __name__ == "__main__":
    main()
