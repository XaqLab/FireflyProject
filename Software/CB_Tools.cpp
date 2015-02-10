// CB_Tools.cpp: implementation of the CCB_Tools class.
//
//////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "CB_Tools.h"


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CCB_Tools::CCB_Tools()
{

}

CCB_Tools::~CCB_Tools()
{

}

//-------------------------------------------------
//takes the name of a Computer Boards' board, and returns the board ID
//number that was assigned to that board by Instacal
int CCB_Tools::GetBoardNum(char *boardName)
{
	int err;
	int board_num;
	char ULBoardName[BOARDNAMELEN];

	//search for the board having name equal to boardName
	for (board_num=0; board_num<20; board_num++)
	{
		err = cbGetBoardName(board_num, ULBoardName);
		if (!err)
		{
			if (stricmp(ULBoardName,boardName) == 0)
				return(board_num);
		}
	}
	return(-1); //no boards found
}

//--------------------------------------------------------------
//takes a board number and chip number and returns the base address
//of the selected 8255 chip.  THis base address is needed to set up the RDX link
int CCB_Tools::Get8255BaseAddr(int boardNum, int chipNum)
{
	int err;
	int BaseAddr;

	err = cbGetConfig(BOARDINFO,boardNum, chipNum, DIBASEADR, &BaseAddr);
	if (err)
		BaseAddr = -1;
	return(BaseAddr);
}
