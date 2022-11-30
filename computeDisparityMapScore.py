import sys
import cv2
from webcolors import rgb_to_name

color_to_code = {
    "brown" : "definitelyWrongOcclusionError",
    "black" : "definitelyWrongUnknown",
    "red" : "definitelyWrongNoFuse",
    "pink" : "maybeWrongFuseColorMismatch",
    "orange" : "uncertainOcclusion",
    "yellow" : "outOfBoundsOcclusion",
    "magenta" : "maybeWrongSegmentOutlier",
    "purple" : "maybeWrongGlobalOutlier",
    "blue" : "maybeRight",
}

def main():
    outputScoreFile = ""
    if len(sys.argv) == 2:
        outputScoreFile = sys.argv[1]
    else:
        print(
            "Usage: {name} [ dispMapScoreFile ]".format(
                name=sys.argv[0]
            )
        )
        exit()
    outputScore = cv2.imread(outputScoreFile, 1)
    outputScore = cv2.cvtColor(outputScore, cv2.COLOR_BGR2RGB)
    cv2.waitKey(0)  # waits until a key is pressed

    rows = outputScore.shape[0]
    cols = outputScore.shape[1]
    numPixels = rows * cols
    numDefinitelyWrong = numMaybeWrong = numMaybeRight =0

    for r in range(0, rows):
        for c in range(0, cols):
            curColor = rgb_to_name(outputScore[r][c])
            code = color_to_code[curColor]
            if code in ["definitelyWrongOcclusionError", "definitelyWrongUnknown", "definitelyWrongNoFuse"]:
                numDefinitelyWrong += 1
            elif code in ["maybeWrongFuseColorMismatch", "maybeWrongSegmentOutlier", "maybeWrongGlobalOutlier"]:
                numMaybeWrong += 1
            elif code in ["maybeRight"]:
                numMaybeRight += 1
    
    finalScore = (numDefinitelyWrong + 0.5 * numMaybeWrong)/numPixels
    print("Final score for {} is: {}".format(outputScoreFile, finalScore))

if __name__ == "__main__":
    main()

