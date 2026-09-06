# Baby Singular Learning Theory

This project is a tiny experimental playground for testing the part of singular learning theory that suggests we may be able to control which algorithm a model learns by carefully choosing its data.

## Build instructions

To run the project locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript
(but not the HTML and CSS files) whenever they change.

## Credits

Built by Samuel Gélineau on top of the [Tensorflow Playground](https://playground.tensorflow.org/).
