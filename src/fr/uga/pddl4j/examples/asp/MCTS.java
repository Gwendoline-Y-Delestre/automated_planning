package fr.uga.pddl4j.examples.asp;

import fr.uga.pddl4j.heuristics.state.StateHeuristic;
import fr.uga.pddl4j.parser.DefaultParsedProblem;
import fr.uga.pddl4j.plan.Plan;
import fr.uga.pddl4j.plan.SequentialPlan;
import fr.uga.pddl4j.planners.AbstractPlanner;
import fr.uga.pddl4j.problem.DefaultProblem;
import fr.uga.pddl4j.problem.Problem;
import fr.uga.pddl4j.problem.State;
import fr.uga.pddl4j.problem.operator.Action;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import picocli.CommandLine;

import fr.uga.pddl4j.planners.Statistics;
import fr.uga.pddl4j.planners.statespace.HSP;

import java.io.*;
import java.util.*;

@CommandLine.Command(name = "MCTS",
    version = "MRW 1.0",
    description = "Solves a specified planning problem using Monte Carlo Tree Search strategy.",
    sortOptions = false,
    mixinStandardHelpOptions = true,
    headerHeading = "Usage:%n",
    synopsisHeading = "%n",
    descriptionHeading = "%nDescription:%n%n",
    parameterListHeading = "%nParameters:%n",
    optionListHeading = "%nOptions:%n")

public class MCTS extends AbstractPlanner {

    /**
     * The class logger.
     */
    private static final Logger LOGGER = LogManager.getLogger(MCTS.class.getName());

    /**
     * The weight of the heuristic.
     */
    private double heuristicWeight = 1;

    /**
     * The name of the heuristic used by the planner.
     */
    private StateHeuristic.Name heuristicName = StateHeuristic.Name.FAST_FORWARD;
    
    /**
     * Returns the weight of the heuristic.
     *
     * @return the weight of the heuristic.
     */
    public final double getHeuristicWeight() {
        return this.heuristicWeight;
    }

    /**
     * Returns the name of the heuristic used by the planner to solve a planning problem.
     *
     * @return the name of the heuristic used by the planner to solve a planning problem.
     */
    public final StateHeuristic.Name getHeuristic() {
        return this.heuristicName;
    }

    /**
     * Instantiates the planning problem from a parsed problem.
     *
     * @param problem the problem to instantiate.
     * @return the instantiated planning problem or null if the problem cannot be
     *         instantiated.
     */
    @Override
    public Problem instantiate(DefaultParsedProblem problem) {
        final Problem pb = new DefaultProblem(problem);
        pb.instantiate();
        return pb;
    }

    /**
     * Extracts a search from a specified node.
     *
     * @param node the node.
     * @param problem the problem.
     * @return the search extracted from the specified node.
     */
    private Plan extractPlan(final Node node, final Problem problem) {
        Node n = node;
        final Plan plan = new SequentialPlan();
      
        while (n.getAction() != -1) {
          
            final Action a = problem.getActions().get(n.getAction());
           
            plan.add(0, a);
            n = n.getParent();
        }
        return plan;
    }

    public static int NB_WALK = 400; // nb of random walks

    public static int LENGTH_WALK = 7; // length of random walk

    public static int MAX_STEPS = 7; // max step during a walk


    /**
     * @param prob the problem for which we want to find applicable actions. 
     * @param node a specific state or configuration in a problem-solving domain. 
     * It is typically used to determine which actions are applicable or valid
     * in that particular state.
     * @return The method is returning a List of applicable actions.
     */
    private List<Action> getActionsPos(Problem prob, Node node) {
        List<Action> actions = prob.getActions();
        List<Action> applicableActions = new ArrayList<>();
        for (Action action : actions)
            if (action.isApplicable(node))
                applicableActions.add(action);
        return applicableActions;
    }

    /**
     * solves a problem using a pure random walk algorithm with a specified heuristic.
     * 
     * @param problem an instance of the Problem class. It represents the
     * problem that needs to be solved.
     * @return The method is returning a Plan object.
     */
    public Plan pureRandomWalksSolver(Problem problem) {
       
        StateHeuristic heuristic = StateHeuristic.getInstance(this.getHeuristic(), problem);

        
        State init = new State(problem.getInitialState());
        Node node = new Node(init, null, -1, 0, 0, heuristic.estimate(init, problem.getGoal()));

        double heuMin = node.getHeuristic();
      
        int counter = 0;
        
        // checks if a counter variable is greater than or equal to a maximum number of steps or if there
        // are no applicable actions for the current node. If either of these conditions is true, it creates 
        // a new node with initial values and resets the counter.
        while (!node.satisfy(problem.getGoal())) {

            if (counter >= MAX_STEPS || getActionsPos(problem, node).isEmpty()) {
                node = new Node(init, null, -1, 0, 0, heuristic.estimate(init, problem.getGoal()));
                counter = 0;
            }
         
            node = pureRandomWalk(problem, node, heuristic);

            if (node.getHeuristic() < heuMin) {
                heuMin = node.getHeuristic();
                counter = 0;
            } else {
                counter++;
            }
        }

      
        return extractPlan(node, problem);
    }

    /**
     * The function randomly selects an action from a list of actions.
     * 
     * @param listActions A list of Action objects.
     * @return The method is returning an Action object.
     */
    private Action SelectAction(List<Action> listActions) {
        Collections.shuffle(listActions);
        return listActions.get(0);
    }

    /**
     * applies a given action to a node's state, creates a new node with the updated
     * state, and calculates the heuristic value for the new node.
     * 
     * @param prob the problem that we are trying to solve. 
     * @param node represents the current node in the search tree.
     * @param action represents the action that is being applied to the current node.
     * @param heuristic used to estimate the cost or distance from a given node to the goal state in
     *  the problem. The estimate provided by the heuristic is used to guide the search algorithm in 
     * finding the optimal solution.
     * @return The method is returning a Node object.
     */
    public Node applyAction(Problem prob, Node node, Action action, StateHeuristic heuristic) {
        State state = new State(node);
        state.apply(action.getConditionalEffects());
        Node enfant = new Node(state, node, prob.getActions().indexOf(action), node.getCost() + 1, node.getDepth() + 1, 0);
        enfant.setHeuristic(heuristic.estimate(enfant, prob.getGoal()));
        return enfant;
    }

    /**
     * performs a random walk on a given problem and returns the node with the minimum heuristic value 
     * encountered during the walk.
     * 
     * @param prob represents the problem that the random walk is trying to solve.
     * @param node represents the current state of the problem. I
     * @param heuristic The heuristic value is an estimate of how close a state is to the goal state.
     * @return The method `pureRandomWalk` returns a `Node` object.
     */
    public Node pureRandomWalk(Problem prob, Node node, StateHeuristic heuristic) {
       
        double heuMin = Double.MAX_VALUE;
      
        Node nodeMin = null;
     
        for (int i = 0; i < NB_WALK; i++) {
            Node nodePrim = node;
            
            for (int j = 1; j < LENGTH_WALK; j++) {
               
                List<Action> ActionList = this.getActionsPos(prob, nodePrim);
                if (ActionList.isEmpty())
                    break;

                
                Action action = SelectAction(ActionList);
                nodePrim = applyAction(prob, nodePrim, action, heuristic);

                if (nodePrim.satisfy(prob.getGoal()))
                    return nodePrim;
            }
         
            if (nodePrim.getHeuristic() < heuMin) {
                heuMin = nodePrim.getHeuristic();
                nodeMin = nodePrim;
            }
        }

    
        return nodeMin == null ? node : nodeMin;
    }

    /**
     * implements the Monte Carlo Tree Search algorithm to solve a given problem and returns a plan
     * if successful.
     * 
     * @param problem represents the problem to be solved. 
     * @return The method is returning a Plan object.
     */
    @Override
    public Plan solve(Problem problem) {
        LOGGER.info("* Starting MTCS search \n");

        final long heure_debut = System.currentTimeMillis();

        final Plan plan = this.pureRandomWalksSolver(problem);

        final long heure_fin = System.currentTimeMillis();

        if (plan != null) {
            LOGGER.info("* MCTS search succeeded *\n");
            this.getStatistics().setTimeToSearch(heure_fin - heure_debut);
        } else {
            LOGGER.info("* MCTS search failed *\n");
        }
        return plan;
    }

    /**
     * takes an AbstractPlanner object, solves the planning problem, retrieves
     * statistics, calculates the total time spent, and returns the time spent and the length of the
     * plan as a string.
     * 
     * @param planner used to solve a planning problem and retrieve the resulting plan
     * and statistics.
     * @return The method is returning a string that contains the total time spent on parsing,
     * encoding, and searching, as well as the length of the plan. The format of the returned string is
     * "TimeSpent,planLength".
     */
    private static String run(AbstractPlanner planner) throws FileNotFoundException {
        Plan p = planner.solve();
        Statistics s = planner.getStatistics();
        double TimeSpent = s.getTimeToParse() + s.getTimeToEncode() + s.getTimeToSearch();
        int planLength = p == null ? 0 : p.size();
        return TimeSpent + "," + planLength;
    }

    /**
     * The main method of the <code>MRW</code> planner.
     *
     * @param args the arguments of the command line.
     */
    public static void main(String[] args) throws IOException {
        try {
            final MCTS mctsPlanner = new MCTS();
            final HSP hspPlanner = new HSP();

            //  creating a CSV file named "resultats.csv" and writing some data into it.
            File resultFile = new File("pddl/resultats.csv");
            BufferedWriter writer = new BufferedWriter(new FileWriter(resultFile));
            writer.write("domain,n_problem,MCTS_temps,MCTS_taille,HSP_temps,HSP_taille");
            writer.newLine();

            List<File> blocks = List.of(new File("pddl/blocks").listFiles());
            List<File> depots= List.of(new File("pddl/depots").listFiles());
            List<File> gripper = List.of(new File("pddl/gripper").listFiles());
            List<File> logistics = List.of(new File("pddl/logistics").listFiles());

            Map<File, List<File>> pddlDomain = new TreeMap<>();
            pddlDomain.put(new File("pddl/blocks_domain.pddl"), blocks);
            pddlDomain.put(new File("pddl/depots_domain.pddl"), depots);
            pddlDomain.put(new File("pddl/gripper_domain.pddl"), gripper);
            pddlDomain.put(new File("pddl/logistics_domain.pddl"), logistics);

            // For each domain file and problem file pair, it sets the domain and problem paths for mctsPlanner 
            // and hspPlanner, runs the planners, and writes the results to a writer. The results
            // include the domain name, problem number, and the results from both planners.
            for(File domainFile : pddlDomain.keySet()) {
                for(File problemFile : pddlDomain.get(domainFile)) {
                    String domainPath = domainFile.getPath();
                    String problemPath = problemFile.getPath();
                    mctsPlanner.setDomain(domainPath);
                    hspPlanner.setDomain(domainPath);
                    mctsPlanner.setProblem(problemPath);
                    hspPlanner.setProblem(problemPath);

                    String mctsResultats = run(mctsPlanner);
                    String hspResultats = run(hspPlanner);

                    String domain = domainFile.getName();
                    domain = domain.substring(7, domainFile.getName().lastIndexOf("."));
                    String problem = String.valueOf(pddlDomain.get(domainFile).indexOf(problemFile) + 1);
                    writer.write(domain + "," + problem + "," + mctsResultats + "," + hspResultats);
                    writer.newLine();
                }
            }

            writer.close();

        } catch (IllegalArgumentException e) {
            LOGGER.fatal(e.getMessage());
        }
    }

}
