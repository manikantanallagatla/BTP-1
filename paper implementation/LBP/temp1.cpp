#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 * ---
 * Hint: You can use the debug stream to print initialTX and initialTY, if Thor seems not follow your orders.
 **/
int main()
{
    int lightX; // the X position of the light of power
    int lightY; // the Y position of the light of power
    int initialTX; // Thor's starting X position
    int initialTY; // Thor's starting Y position
    cin >> lightX >> lightY >> initialTX >> initialTY; cin.ignore();

    // game loop
    while (1) {
        int remainingTurns; // The remaining amount of turns Thor can move. Do not remove this line.
        cin >> remainingTurns; cin.ignore();

        // A single line providing the move to be made: N NE E SE S SW W or NW
        if(initialTX<lightX and initialTY<lightY){
            initialTX++;
            initialTY++;
            cout<<"NE"<<endl;
        }
        if(initialTX<lightX and initialTY>lightY){
            cout<<"SE"<<endl;
            initialTX++;
            initialTY--;
        }
        if(initialTX>lightX and initialTY<lightY){
            cout<<"SW"<<endl;

            initialTX--;
            initialTY++;
        }
        if(initialTX>lightX and initialTY>lightY){
            cout<<"SE"<<endl;
            initialTX--;
            initialTY--;
        }
        if(initialTX==lightX and initialTY<lightY){
            cout<<"N"<<endl;
            // initialTX++;
            initialTY--;
        }
        if(initialTX==lightX and initialTY>lightY){
            cout<<"S"<<endl;
            // initialTX++;
            initialTY++;
        }
        if(initialTX<lightX and initialTY==lightY){
            cout<<"E"<<endl;

            initialTX++;
            // initialTY++;
        }
        if(initialTX>lightX and initialTY==lightY){
            cout<<"W"<<endl;

            initialTX--;
            // initialTY++;
        }
    }
}