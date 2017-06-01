// Image Correlation Genetic Algorithm (GA) Header Class

// notes: uses fitness proportionate (roulette) selection
// with conditional replacement of parents with offspring

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2012 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#ifndef GA_H
#define GA_H

/******************************************************************************/

#include <cv.h>   		// open cv general include file
#include <highgui.h>	// open cv GUI include file
#include <iostream>		// standard C++ I/O
#include <bitset>	    // C++ bitset

using namespace cv; // OpenCV API is in the C++ "cv" namespace

/******************************************************************************/

// define a Gene type that stores a 2D point and its associated fitness

typedef struct {

    unsigned short x;         // use 16-bit to minimise redundant bits and maximise
    unsigned short y;         // effective use of mutation on representation
    double fitness;

} Gene_t;

#define GA_BAD_FITNESS 0

/******************************************************************************/

class correlationGA
{
    public:
        // popSize = population size
        // crossover = probability of crossover (0->1)
        // mutation = probability of mutation (0->1)

        correlationGA(int popSize, double crossover, double mutation,
                       Mat img, Mat roi) {

            // get image and correlation template

            image = img.clone();
            correlation_template = roi.clone();
            populationSize = popSize;

            // get number to be cross'd over and mutated in each population cycle

            crossoverCount = (int) floor(populationSize * crossover);
            mutationCount = (int) floor(populationSize * mutation);

            random = RNG(getCPUTickCount()); // initialise random with time

            // initialise and evaulate fitness of first population

            Gene_t tmp;

            for (int i = 0; i < populationSize; i++)
            {
                    tmp.x = random(img.cols);
                    tmp.y = random(img.rows);
                    tmp.fitness = fitness(tmp);
                    population.push_back(tmp);

            }

        }

        /***********************************************************************/

        void createNextGeneration(){

            double sumFitness = 0;
            vector<Gene_t> selection_list;
            vector<Gene_t> next_population;

            // calculate total fitness

            for (int i = 0; i < populationSize; i++)
            {
                sumFitness += population[i].fitness;
            }

            // insert item into list fitness / sumFitness times

            for (int i = 0; i < populationSize; i++)
            {
                double fitness_prop_i = (population[i].fitness / sumFitness) * 100;
                while (fitness_prop_i > 0)
                {
                    selection_list.push_back(population[i]);
                    fitness_prop_i -= 1.0;
                }

            }

            // randomly select genes from selection list and copy to next
            // generation

            for (int i = 0; i < populationSize; i++)
            {
                next_population.push_back(selection_list[random(selection_list.size())]);
            }

            // perform cross-over

            for (int i = 0; i < crossoverCount; i++)
            {
                int first = random(populationSize);
                int second = random(populationSize);

                Gene_t firstOffspring = crossover(next_population[first], next_population[second]);
                Gene_t secondOffspring = crossover(next_population[second], next_population[first]);

                firstOffspring.fitness = fitness(firstOffspring);
                secondOffspring.fitness = fitness(secondOffspring);

                // replace parent with offspring if fitter (not always what is done but we will use
                // this here)

                if (firstOffspring.fitness > next_population[first].fitness)
                {
                    next_population[first] = firstOffspring;
                }
                if (secondOffspring.fitness > next_population[second].fitness)
                {
                    next_population[second] = secondOffspring;
                }

            }

            // perform mutation (in place)

            for (int i = 0; i < mutationCount; i++)
            {
                int mutated = random(populationSize);

                mutation(next_population[mutated]);
                next_population[mutated].fitness = fitness(next_population[mutated]);
            }

            // copy next population to current population

            population = next_population;

        }


        /***********************************************************************/

        // return best performning gene from current population

        Gene_t returnMaximal() {

            Gene_t maxG;
            maxG.fitness = GA_BAD_FITNESS;

            for (int i = 0; i < populationSize; i++)
            {
                if (population[i].fitness > maxG.fitness)
                {
                    maxG = population[i];
                }
            }

            return maxG;
        }

        /***********************************************************************/

        virtual ~correlationGA() {}
    protected:
        int crossoverCount;
        int mutationCount;
        RNG random;

        /***********************************************************************/

        double fitness(Gene_t gene) {

            Mat result;
            Mat tmp = image.clone();
            double minVal;

            // check for out of range locations

            if ((gene.x < 0) || (gene.x >= image.cols))
                return GA_BAD_FITNESS;
            if ((gene.y < 0) || (gene.y >= image.rows))
                return GA_BAD_FITNESS;
            if ((gene.x + correlation_template.cols) >= image.cols)
                return GA_BAD_FITNESS;
            if ((gene.y + correlation_template.rows) >= image.rows)
                return GA_BAD_FITNESS;

            // otherwise do correlation of template against image at location
            // specified by the gene using squared difference

            matchTemplate(image(Rect(gene.x, gene.y,
                          correlation_template.cols, correlation_template.rows)),
                          correlation_template, result, CV_TM_SQDIFF);

            // draw current fitness evaulation

            rectangle(tmp, Rect(gene.x, gene.y,
                          correlation_template.cols, correlation_template.rows), Scalar(0,255,0), 2);

            imshow("Fitness Evaluation", tmp);
            waitKey(5);

            //return minimum squared difference from the matchTemplate() result

            minMaxLoc(result, &minVal, NULL, NULL, NULL);

            return (1.0 / minVal); // return high fitness (1 / difference) for good matches

        }

        /***********************************************************************/

        Gene_t crossover(Gene_t A, Gene_t B)
        {

            Gene_t tmp;

            // gene is a 2D point so cross-over is simple
            // (see C++ bitset() class for implementing more complex approaches
            //  at a binary level)

            tmp.x = A.x;
            tmp.y = B.y;
            tmp.fitness = 0; // for sanity

            return tmp;

        }

        /***********************************************************************/

        void mutation(Gene_t &A) {

            int xy = random(2); // choose location (0=x, 1=y)

            // see C++ Std. Lib. bitset() template class for details

            if (xy == 0){ // randomly swap a bit in either x or y
               std::bitset<sizeof(unsigned short)> bits =  A.x;
               bits.flip((size_t) random(bits.size()));
               A.x = (unsigned short) bits.to_ulong();
            } else {
               std::bitset<sizeof(unsigned short)> bits =  A.y;
               bits.flip((size_t) random(bits.size()));
               A.y = (unsigned short) bits.to_ulong();
            }
        }

        /***********************************************************************/

    private:
        vector<Gene_t> population;
        int populationSize;
        Mat image;
        Mat correlation_template;
};

#endif // GA_H
