## Task Description

In this year the aim of our group project will be the construction (digital) of a mobility dashboard with modern technologies from Python/HTML/JavaScript or any other solution you can image to be used to make mobility data vivid and understandable.
We provide you initially some example recording data, recorded by smartphones and by trackers. You'll have to implement your prefered dashboard by using all learned skills of data analysis of this course.

## The Dashboard Constraints

1. A good overview about the track statistics that display the characteristic metrics of all GPS-trackings like the average velocity/acceleration/track-lenghts/modality-share and others that are important in your opinion.
2. A map that visualizes the tracks in different colors, shapes and with a legend.
3. A classification algorithm that classifies the track data according to its modality based on the existing information. Sometimes hdop,gyroscope or magnetometric data is availiable sometimes not. GPS data will be availiable all time. You need to integrate the modality information in your dashboard as well.
4. It's on you to decide whether you also include std.-deviations and other stochastic metrics and in which color theme you design the board. It should display the stats concise and clear to the audience.
5. For the experts: Implement a animation that repeats, which means you see for suitable cases the change of the data over time (e.g. day-wise, week-wise, month-wise from a cerain time till now) (+ additional plus points).
6. For the experts: Implement it dynamically. This means we can pass in data and the dashboards updates on its dynamical by updating all stats and probably highlighting changes (+ additional plus points).

## The Testing at the presentation day

We'll provide you some new data of the same format at the presentation day. The data will be recorded in the next days so you won't see it before the presentation day. We'll have already analyzed the data in a first shot to know how it should look like. We additionally already know the modality of the data, but it won't be given to you. Based on the new data we'll try out your board during the presentation and where you have to show the new stats and where your system will have to the new tracks modality on its own. We'll all then see the results of your dashboard implementation and your classification algorithm. We will then compare it to the already existing modality information of our track side by side.
