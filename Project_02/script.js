const words = [
    "Cloud Engineer",
    "DevOps Engineer",
    "Entrepreneur",
    "AI Enthusiast",
    "Founder"
];

let wordIndex = 0;
let charIndex = 0;
let isDeleting = false;

const typing = document.getElementById("typing");

function typeEffect() {

    let currentWord = words[wordIndex];

    if (!isDeleting) {

        typing.textContent =
            currentWord.substring(0, charIndex + 1);

        charIndex++;

        if (charIndex === currentWord.length) {
            isDeleting = true;
            setTimeout(typeEffect, 1500);
            return;
        }

    } else {

        typing.textContent =
            currentWord.substring(0, charIndex - 1);

        charIndex--;

        if (charIndex === 0) {
            isDeleting = false;
            wordIndex =
                (wordIndex + 1) % words.length;
        }
    }

    setTimeout(typeEffect, isDeleting ? 50 : 100);
}

typeEffect();
particlesJS("particles-js",{

particles:{
number:{
value:80
},

size:{
value:3
},

move:{
speed:2
},

line_linked:{
enable:true
}
}

});
