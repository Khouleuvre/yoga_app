import argparse

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Video classification program")

    # Add the arguments
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path to the input video file')
    parser.add_argument('--display', action='store_true', 
                        help='Display the video while processing')
    parser.add_argument('--output', type=str, 
                        help='Path to the output video file')

    # Execute the parse_args() method
    args = parser.parse_args()

    print(f"Input file: {args.input}")
    if args.display:
        print("Display is turned on")
    if args.output:
        print(f"Output file: {args.output}")
