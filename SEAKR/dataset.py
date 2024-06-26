from typing import Any, List, Dict
from abc import ABC, abstractmethod
import json



class Dataset(ABC):
    def __init__(self, n_shot) -> None:
        super().__init__()
        self.n_shot = min(n_shot, len(self.examples))
        self.prefix = "\n\n".join([self.demo_template(**ex) for ex in self.examples[:self.n_shot]])

    def demo_template(self, question: str, cot: List[str], answer: str):
        steps_text = " ".join([f"{c.replace('.', ' ').strip()}." for i, c in enumerate(cot)])
        return f"Question: {question}\nAnswer: {steps_text} So the answer is {answer}."
    
    def __call__(self, question: str, cot_steps: List[str]=None, docs: List[str]=None):
        input_prompt = f"Answer in the same format as before. \nQuestion: {question}\nAnswer: "
        if cot_steps:
            input_prompt += " ".join([f"{c.replace('.', ' '.strip())}." for c in cot_steps])
        if docs:
            docs_text = "Context: "
            docs_text += "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
            # docs_text += "\nAnswer in the same format as before."
            input_prompt = f"{docs_text}\n{input_prompt}"
        return self.prefix + "\n\n" + input_prompt

    @abstractmethod
    def load_data(cls, file_path: str):
        pass
        

def get_dataset(dataset_name: str, n_shot: int) -> Dataset:
    match dataset_name.lower():
        case "twowikihop":
            return TwoWikiHop(n_shot)
        case "hotpotqa":
            return HotpotQA(n_shot)
        case "iirc":
            return IIRC(n_shot)
        case "natural_questions":
            dataset_obj = SingleQA(n_shot)
            setattr(dataset_obj, 'default_data_path', f"./data/singlehop_data/processed_nq.json")
            return dataset_obj
        case "triviaqa":
            dataset_obj = SingleQA(n_shot)
            setattr(dataset_obj, 'default_data_path', f"./data/singlehop_data/processed_tq.json")
            return dataset_obj
        case "squad":
            dataset_obj = SingleQA(n_shot)
            setattr(dataset_obj, 'default_data_path', f"./data/singlehop_data/processed_sq.json")
            return dataset_obj
        case _:
            raise NotImplementedError


class TwoWikiHop(Dataset):
    examples: List[Dict] = [
        {
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': [
                "Martin Hodge was born on 4 February 1959.",
                "Ivania Martinich was born on 25 July 1995.",
                "Thus, 4 February 1959 is earlier than 25 July 1995 and Martin Hodge was born first."
            ],
            'answer': "Martin Hodge",
        },
        {
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': [
                "The film Hypocrite was directed by Miguel Morayta.",
                "Miguel Morayta died on 19 June 2013."
            ],
            'answer': "19 June 2013",
        },
        {
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': [
                "Kurram Garhi is located in the country of Pakistan.",
                "Trojkrsti is located in the country of Republic of Macedonia.",
                "Thus, they are not in the same country."
            ],
            'answer': "no",
        },
        {
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': [
                "Coolie No. 1 (1995 film) was directed by David Dhawan.",
                "The Sensational Trial was directed by Karl Freund.",
                "David Dhawan's nationality is India.",
                "Karl Freund's nationality is Germany.",
                "Thus, they do not have the same nationality."
            ],
            'answer': "no",
        },
        {
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': [
                "Boraqchin is married to Ögedei Khan.",
                "Ögedei Khan's father is Genghis Khan.",
                "Thus, Boraqchin's father-in-law is Genghis Khan."
            ],
            'answer': "Genghis Khan",
        },
        {
            'question': "When did the director of film Laughter In Hell die?",
            'cot': [
                "The film Laughter In Hell was directed by Edward L. Cahn.",
                "Edward L. Cahn died on August 25, 1963."
            ],
            'answer': "August 25, 1963",
        },
        {
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': [
                "Krishna Shah has a child named Rudra Shah.",
                "Rudra Shah has a child named Prithvipati Shah.",
                "Thus, Krishna Shah has a grandchild named Prithvipati Shah."
            ],
            'answer': "Prithvipati Shah",
        },
        {
            'question': "Where did the director of film Maddalena (1954 Film) die?",
            'cot': [
                "The film Maddalena is directed by Augusto Genina.",
                "Augusto Genina died in Rome."
            ],
            'answer': "Rome",
        },
        {
            'question': "What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?",
            'cot': [
                "The mother of Grand Duke Alexei Alexandrovich of Russia is Maria Alexandrovna.",
                "Maria Alexandrovna died from tuberculosis."
            ],
            'answer': "tuberculosis",
        },
        {
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': [
                "The film Twenty Plus Two was directed by Joseph M. Newman.",
                "The Gal Who Took the West was directed by Frederick de Cordova.",
                "Joseph M. Newman died on January 23, 2006.",
                "Fred de Cordova died on September 15, 2001.",
                "Thus, January 23, 2006 is later than September 15, 2001, and the person to die later from the two is Twenty Plus Two."
            ],
            'answer': "Twenty Plus Two",
        },
    ]
    
    @classmethod
    def load_data(cls, file_path: str="./data/multihop_data/2wikimultihopqa/dev.json"):
        dataset = []
        with open(file_path, 'r') as fin:
            js = json.load(fin)
            print(len(js))
            for example in js:
                qid = example['_id']
                question = example['question']
                ans = example['answer']
                ans_id = example['answer_id']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                })
        return dataset
        

class HotpotQA(Dataset):
    examples: List[Dict] = [
    {
        'question': "Jeremy Theobald and Christopher Nolan share what profession?",
        'cot': [
            "Jeremy Theobald is an actor and producer.",
            "Christopher Nolan is a director, producer, and screenwriter.",
            "Therefore, they both share the profession of being a producer."
        ],
        'answer': "producer"
    },
    {
        'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
        'cot': [
            "Brian Patrick Butler directed the film The Phantom Hour.",
            "The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari.",
            "Of these Nosferatu was directed by F.W. Murnau."
        ],
        'answer': "The Phantom Hour"
    },
    {
        'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
        'cot': [
            "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988.",
            "The number of episodes Reply 1988 has is 20."
        ],
        'answer': "20"
    },
    {
        'question': "Were Lonny and Allure both founded in the 1990s?",
        'cot': [
            "Lonny (magazine) was founded in 2009.",
            "Allure (magazine) was founded in 1991.",
            "Thus, of the two, only Allure was founded in 1990s."
        ],
        'answer': "no"
    },
    {
        'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in 'The Right Stuff'?",
        'cot': [
            "The actor who played astronaut Alan Shepard in 'The Right Stuff' is Scott Glenn.",
            "The movie Vertical Limit also starred Scott Glenn."
        ],
        'answer': "Scott Glenn"
    },
    {
        'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
        'cot': [
            "Lake Wales Medical Center is located in the city of Lake Wales, Polk County, Florida.",
            "The population of Lake Wales in 2014 was 15,140."
        ],
        'answer': "15,140"
    },
    {
        'question': "Who was born first? Jan de Bont or Raoul Walsh?",
        'cot': [
            "Jan de Bont was born on 22 October 1943.",
            "Raoul Walsh was born on March 11, 1887.",
            "Thus, Raoul Walsh was born the first."
        ],
        'answer': "Raoul Walsh"
    },
    {
        'question': "In what country was Lost Gravity manufactured?",
        'cot': [
            "The Lost Gravity (roller coaster) was manufactured by Mack Rides.",
            "Mack Rides is a German company."
        ],
        'answer': "Germany"
    },
    {
        'question': "Which of the following had a debut album entitled 'We Have an Emergency': Hot Hot Heat or The Operation M.D.?",
        'cot': [
            "The debut album of the band 'Hot Hot Heat' was 'Make Up the Breakdown'.",
            "The debut album of the band 'The Operation M.D.' was 'We Have an Emergency'."
        ],
        'answer': "The Operation M.D."
    },
    {
        'question': "How many awards did the 'A Girl Like Me' singer win at the American Music Awards of 2012?",
        'cot': [
            "The singer of 'A Girl Like Me' is Rihanna.",
            "In the American Music Awards of 2012, Rihanna won one award."
        ],
        'answer': "one"
    },
    {
        'question': "The actor that stars as Joe Proctor on the series 'Power' also played a character on 'Entourage' that has what last name?",
        'cot': [
            "The actor that stars as Joe Proctor on the series 'Power' is Jerry Ferrara.",
            "Jerry Ferrara also played a character on Entourage named Turtle Assante.",
            "Thus, Turtle Assante's last name is Assante."
        ],
        'answer': "Assante"
    },
    {
        'question': "In which country did this Australian who was detained in Guantanamo Bay detention camp and published 'Guantanamo: My Journey' receive para-military training?",
        'cot': [
            "The Australian who was detained in Guantanamo Bay detention camp and published 'Guantanamo: My Journey' is David Hicks.",
            "David Hicks received his para-military training in Afghanistan."
        ],
        'answer': "Afghanistan"
    },
    {
        'question': "Does The Border Surrender or Unsane have more members?",
        'cot': [
            "The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin.",
            "That is, it has 4 members.",
            "Unsane is a trio of 3 members.",
            "Thus, The Border Surrender has more members."
        ],
        'answer': "The Border Surrender"
    },
    {
        'question': "Which band formed first, Sponge Cola or Hurricane No. 1?",
        'cot': [
            "Sponge Cola band was formed in 1998.",
            "Hurricane No. 1 was formed in 1996.",
            "Thus, Hurricane No. 1 band formed the first."
        ],
        'answer': "Hurricane No. 1"
    },
    {
        'question': "James Paris Lee is best known for investing the Lee-Metford rifle and another rifle often referred to by what acronymn?",
        'cot': [
            "James Paris Lee is best known for investing the Lee-Metford rifle and Lee-Enfield series of rifles.",
            "Lee-Enfield is often referred to by the acronym of SMLE."
        ],
        'answer': "SMLE"
    },
    {
        'question': "Who was born first, James D Grant, who uses the pen name of Lee Child, or Bernhard Schlink?",
        'cot': [
            "James D Grant, who uses the pen name of Lee Child, was born in 1954.",
            "Bernhard Schlink was born in 1944.",
            "Thus, Bernhard Schlink was born first."
        ],
        'answer': "Bernhard Schlink"
    },
    {
        'question': "Which American neo-noir science fiction has Pierce Gagnon starred?",
        'cot': [
            "Pierce Gagnon has starred in One Tree Hill, Looper, Wish I Was Here and Extant.",
            "Of these, Looper is an American neo-noir science fiction."
        ],
        'answer': "Looper"
    },
    {
        'question': "What year did Edburga of Minster-in-Thanet's father die?",
        'cot': [
            "The father of Edburga of Minster-in-Thanet is King Centwine.",
            "Centwine died after 685."
        ],
        'answer': "after 685"
    },
    {
        'question': "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?",
        'cot': [
            "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges.",
            "Nobody Loves You was written by John Lennon on Walls and Bridges album."
        ],
        'answer': "Walls and Bridges"
    },
    {
        'question': "Who is older Jeremy Horn or Renato Sobral?",
        'cot': [
            "Jeremy Horn was born on August 25, 1975.",
            "Renato Sobral was born on September 7, 1975.",
            "Thus, Jeremy Horn is older."
        ],
        'answer': "Jeremy Horn"
    }
]   
    
    @classmethod
    def load_data(cls, file_path: str= "data/multihop_data/hotpotqa/hotpotqa-dev.json"):
        dataset = []
        with open(file_path, "r") as fin:
            js = json.load(fin)
            for example in js:
                qid = example["_id"]
                question = example["question"]
                answer = example['answer']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                })
        return dataset


class IIRC(Dataset):
    examples = [
    {
        "question": "What is the age difference between the kicker and the quarterback for the Chargers?",
        "cot": [
            "The kicker for the Chargers is Nate Kaeding.",
            "The quarterback (QB) for the Chargers is Philip Rivers.",
            "Nate Kaeding was born in the year 1982.",
            "Philip Rivers was born in the year 1981.",
            "Thus, the age difference between them is of 1 year."
        ],
        "answer": "1"
    },
    {
        "question": "How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
        "cot": [
            "The ship that took the battalion from New South Wales to Ceylon is General Hewitt.",
            "General Hewitt was launched in Calcutta in 1811.",
            "General Hewitt was sold for a hulk or to be broken up in 1864.",
            "So she served for a total of 1864 - 1811 = 53 years."
        ],
        "answer": "53"
    },
    {
        "question": "What year was the theatre that held the 2016 NFL Draft built?",
        "cot": [
            "The theatre that held the 2016 NFL Draft is Auditorium Theatre.",
            "The Auditorium Theatre was built in 1889."
        ],
        "answer": "1889"
    },
    {
        "question": "How long had Milan been established by the year that Nava returned there as a reserve in the first team's defense?",
        "cot": [
            "Nava returned to Milan as a reserve in the first team's defense in the year 1990.",
            "Milan had been established in the year 1899.",
            "Thus, Milan had been established for 1990 - 1899 = 91 years when Nava returned to Milan as a reserve in the first team's defense."
        ],
        "answer": "91"
    },
    {
        "question": "When was the town Scott was born in founded?",
        "cot": [
            "Scott was born in the town of Cooksville, Illinois.",
            "Cooksville was founded in the year 1882."
        ],
        "answer": "1882"
    },
    {
        "question": "In what country did Wright leave the French privateers?",
        "cot": [
            "Wright left the French privateers in Bluefield's river.",
            "Bluefields is the capital of the South Caribbean Autonomous Region (RAAS) in the country of Nicaragua."
        ],
        "answer": "Nicaragua"
    },
    {
        "question": "Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
        "cot": [
            "Dr. Hibbert fashioned his hair after Mr. T from The A-Team.",
            "Mr T.'s birthname is Lawrence Tureaud."
        ],
        "answer": "Lawrence Tureaud"
    },
    {
        "question": "How many people attended the conference held near Berlin in January 1942?",
        "cot": [
            "The conference held near Berlin in January 1942 is Wannsee Conference.",
            "Wannsee Conference was attended by 15 people."
        ],
        "answer": "15"
    },
    {
        "question": "When did the country Ottwalt went into exile in founded?",
        "cot": [
            "Ottwalt went into exile in the country of Denmark.",
            "Denmark has been inhabited since around 12,500 BC."
        ],
        "answer": "12,500 BC"
    },
    {
        "question": "When was the J2 club Uki played for in 2001 founded?",
        "cot": [
            "The J2 club that Uki played for is Montedio Yamagata.",
            "Montedio Yamagata was founded in 1984."
        ],
        "answer": "1984"
    },
    {
        "question": "When was the person who produced A Little Ain't Enough born?",
        "cot": [
            "A Little Ain't Enough was produced by Bob Rock.",
            "Bob Rock was born on April 19, 1954."
        ],
        "answer": "April 19, 1954"
    },
    {
        "question": "Which of the schools Fiser is affiliated with was founded first?",
        "cot": [
            "The schools that Fiser is affiliated with (1) Academy of Music, University of Zagreb (2) Mozarteum University of Salzburg (3) Croatian Music Institute orchestra.",
            "Academy of Music, University of Zagreb was founded in the year 1829.",
            "Mozarteum University of Salzburg was founded in the year 1841.",
            "Croatian Music Institute was founded in the year 1827.",
            "Thus, the school founded earliest of these is Croatian Music Institute."
        ],
        "answer": "Croatian Music Institute"
    },
    {
        "question": "How many casualties were there at the battle that Dearing fought at under Jubal Early?",
        "cot": [
            "Under Jubal Early, Dearing fought the First Battle of Bull Run.",
            "First Battle of Bull Run has 460 union casualties and 387 confederate casualties.",
            "Thus, in total the First Battle of Bull Run had 460 + 387 = 847 casualties."
        ],
        "answer": "847"
    },
    {
        "question": "Which of the two congregations which provided leadership to the Pilgrims was founded first?",
        "cot": [
            "The congregations which provided leadership to the Pilgrims are Brownists and Separatist Puritans.",
            "Brownist was founded in 1581.",
            "The Separatist Puritans was founded in 1640.",
            "Thus, Brownist was founded first."
        ],
        "answer": "Brownist"
    },
    {
        "question": "How long had the Rock and Roll Hall of Fame been open when the band was inducted into it?",
        "cot": [
            "The band was inducted into Rock and Roll Hall of Fame in the year 2017.",
            "Rock and Roll Hall of Fame was established in the year of 1983.",
            "Thus, Rock and Roll Hall of Fame been open for 2017 - 1983 = 34 years when the band was inducted into it."
        ],
        "answer": "34"
    },
    {
        "question": "Did the Lord Sewer who was appointed at the 1509 coronation live longer than his king?",
        "cot": [
            "Lord Sewer who was appointed at the 1509 coronation was Robert Radcliffe, 1st Earl of Sussex.",
            "Lord Sever's king in 1509 was Henry VIII of England.",
            "Robert Radcliffe, 1st Earl of Sussex was born in the year 1483, and died in the year 1542.",
            "So Robert lived for 1542 - 1483 = 59 years.",
            "Henry VIII of England was born in the year 1491 and died in the year 1547.",
            "So Henry VIII lived for 1547 - 1491 = 56 years.",
            "Thus, Robert Radcliffe lived longer than Henry VIII."
        ],
        "answer": "yes"
    },
    {
        "question": "When was the place near where Manuchar was defeated by Qvarqvare established?",
        "cot": [
            "Manuchar was defeated by Qvarqvare near Erzurum.",
            "Erzurum was founded during the Urartian period."
        ],
        "answer": "Urartian period"
    },
    {
        "question": "What year was the man who implemented the 46 calendar reform born?",
        "cot": [
            "The man who implemented the 46 calendar reform is Julius Caesar.",
            "Julius Caesar was born in the year 100 BC."
        ],
        "answer": "100 BC"
    },
    {
        "question": "How many years after the first recorded Tommy John surgery did Scott Baker undergo his?",
        "cot": [
            "The first recorded Tommy John surgery happened when it was invented in the year 1974.",
            "Scott Baker underwent Tommy John surgery in the year 2012.",
            "Thus, Scott Baker underwent Tommy John surgery 2012 - 1974 = 38 years after it was first recorded."
        ],
        "answer": "38"
    },
    {
        "question": "Which was the older of the two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK?",
        "cot": [
            "The two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK are Koudas and Matzourakis.",
            "Koudas was born on 23 November 1946.",
            "Matzourakis was born on 6 June 1949.",
            "Thus, the older person among the two is Koudas."
        ],
        "answer": "Koudas"
    }
]
    
    @classmethod
    def load_data(cls, file_path: str= "./data/multihop_data/iirc/iirc_train_dev/dev.json"):
        dataset = []
        with open(file_path, "r") as fin:
            js = json.load(fin)
            for tmp in js:
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']

                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]
                    
                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })
        return dataset
    


class SingleQA(Dataset):
    examples: List[Dict] = [
    {
        "question": "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?",
        "cot": [
            "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges.",
            "Nobody Loves You was written by John Lennon on Walls and Bridges album."
        ],
        "answer": "Walls and Bridges"
    },
    {
        "question": "What is known as the Kingdom and has National Route 13 stretching towards its border?",
        "cot": [
            "Cambodia is officially known as the Kingdom of Cambodia.",
            "National Route 13 stretches towards the border to Cambodia."
        ],
        "answer": "Cambodia"
    },
    {
        "question": "Jeremy Theobald and Christopher Nolan share what profession?",
        "cot": [
            "Jeremy Theobald is an actor and producer.",
            "Christopher Nolan is a director, producer, and screenwriter.",
            "Therefore, they both share the profession of being a producer."
        ],
        "answer": "producer"
    },
    {
        "question": "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
        "cot": [
            "Brian Patrick Butler directed the film The Phantom Hour.",
            "The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari.",
            "Of these Nosferatu was directed by F.W. Murnau."
        ],
        "answer": "The Phantom Hour"
    },
    {
        "question": "Vertical Limit stars which actor who also played astronaut Alan Shepard in 'The Right Stuff'?",
        "cot": [
            "The actor who played astronaut Alan Shepard in 'The Right Stuff' is Scott Glenn.",
            "The movie Vertical Limit also starred Scott Glenn."
        ],
        "answer": "Scott Glenn"
    },
    {
        "question": "Which car, produced by Ferrari from 1962 to 1964 for homologation into the FIA's Group 3 Grand Touring Car category inspired the Vandenbrink GTO?",
        "cot": [
            "The car produced by Ferrari from 1962 to 1964 for homologation into the FIA's Group 3 Grand Touring Car category is the Ferrari 250 GTO.",
            "The Ferrari 250 GTO also inspired the Vandenbrink GTO's styling."
        ],
        "answer": "Ferrari 250 GTO"
    },
    {
        "question": "The actor that stars as Joe Proctor on the series 'Power' also played a character on 'Entourage' that has what last name?",
        "cot": [
            "The actor that stars as Joe Proctor on the series 'Power' is Jerry Ferrara.",
            "Jerry Ferrara also played a character on Entourage named Turtle Assante.",
            "Thus, Turtle Assante's last name is Assante."
        ],
        "answer": "Assante"
    },
    {
        "question": "Who is older Jeremy Horn or Renato Sobral?",
        "cot": [
            "Jeremy Horn was born on August 25, 1975.",
            "Renato Sobral was born on September 7, 1975.",
            "Thus, Jeremy Horn is older."
        ],
        "answer": "Jeremy Horn"
    },
    {
        "question": "In what country was Lost Gravity manufactured?",
        "cot": [
            "The Lost Gravity (roller coaster) was manufactured by Mack Rides.",
            "Mack Rides is a German company."
        ],
        "answer": "Germany"
    },
    {
        "question": "Who was married to a founding member of Nirvana?",
        "cot": [
            "The founding member of Nirvana is Kurt Cobain.",
            "Kurt Cobain was married to Courtney Love."
        ],
        "answer": "Courtney Love"
    },
    {
        "question": "What was the 2014 population of the city where Lake Wales Medical Center is located?",
        "cot": [
            "Lake Wales Medical Center is located in the city of Lake Wales, Florida.",
            "The population of Lake Wales in 2014 was 15,140."
        ],
        "answer": "15,140"
    },
    {
        "question": "Which American neo-noir science fiction has Pierce Gagnon starred?",
        "cot": [
            "Pierce Gagnon has starred in One Tree Hill, Looper, Wish I Was Here and Extant.",
            "Of these, Looper is an American neo-noir science fiction."
        ],
        "answer": "Looper"
    },
    {
        "question": "What Scottish nobleman was the subject of a 1934 British short documentary and was the first man to fly over Mount Everest?",
        "cot": [
            "The Scottish nobleman that was the subject of a 1934 British short documentary was Douglas Douglas-Hamilton.",
            "Douglas Douglas-Hamilton was also the first man to fly over Mount Everest."
        ],
        "answer": "Douglas Douglas-Hamilton"
    },
    {
        "question": "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
        "cot": [
            "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988.",
            "The number of episodes Reply 1988 has is 20."
        ],
        "answer": "20"
    },
    {
        "question": "Does The Border Surrender or Unsane have more members?",
        "cot": [
            "The Border Surrender band has the following members: Keith Austin, Simon Shields, Johnny Manning, and Mark Austin.",
            "That is, it has 4 members.",
            "Unsane is a trio of 3 members.",
            "Thus, The Border Surrender has more members."
        ],
        "answer": "The Border Surrender"
    },
    {
        "question": "Who was born first, James D Grant, who uses the pen name of Lee Child, or Bernhard Schlink?",
        "cot": [
            "James D Grant, who uses the pen name of Lee Child, was born in 1954.",
            "Bernhard Schlink was born in 1944.",
            "Thus, Bernhard Schlink was born first."
        ],
        "answer": "Bernhard Schlink"
    },
    {
        "question": "Which band formed first, Sponge Cola or Hurricane No. 1?",
        "cot": [
            "Sponge Cola band was formed in 1998.",
            "Hurricane No. 1 was formed in 1996.",
            "Thus, Hurricane No. 1 band formed first."
        ],
        "answer": "Hurricane No. 1"
    },
    {
        "question": "What are the nationalities of Leonid Khachiyan and Sofia Kovalevskaya?",
        "cot": [
            "The nationality of Leonid Khachiyan is Russian.",
            "The nationality of Sofia Kovalevskaya is Russian."
        ],
        "answer": "Russian"
    },
    {
        "question": "Which former Canadian Professional Ice Hockey Player endorses MLX Skates?",
        "cot": [
            "MLX Skates is endorsed by Mario Lemieux.",
            "Mario Lemieux is a former Canadian Professional Ice Hockey Player."
        ],
        "answer": "Mario Lemieux"
    },
    {
        "question": "In which country did this Australian who was detained in Guantanamo Bay detention camp and published 'Guantanamo: My Journey' receive para-military training?",
        "cot": [
            "The Australian who was detained in Guantanamo Bay detention camp and published 'Guantanamo: My Journey' is David Hicks.",
            "David Hicks received his para-military training in Afghanistan."
        ],
        "answer": "Afghanistan"
    },
    {
        "question": "What year did Edburga of Minster-in-Thanet's father die?",
        "cot": [
            "The father of Edburga of Minster-in-Thanet is King Centwine.",
            "Centwine died after 685."
        ],
        "answer": "after 685"
    },
    {
        "question": "In which state of Australia will you find the themed lands Ocean parade and DreamWorks Experience both within the Dreamworld theme park complex on the Gold Coast?",
        "cot": [
            "The themed land of Ocean Parade is in the state of Queensland in Australia.",
            "The themed land of The DreamWorks Experience is in the state of Queensland in Australia.",
            "Thus, both Ocean Parade and The DreamWorks Experience are in the state of Queensland."
        ],
        "answer": "Queensland"
    },
    {
        "question": "Were Lonny and Allure both founded in the 1990s?",
        "cot": [
            "Lonny (magazine) was founded in 2009.",
            "Allure (magazine) was founded in 1991.",
            "Thus, of the two, only Allure was founded in 1990s."
        ],
        "answer": "no"
    },
    {
        "question": "James Paris Lee is best known for investing the Lee-Metford rifle and another rifle often referred to by what acronym?",
        "cot": [
            "James Paris Lee is best known for investing the Lee-Metford rifle and Lee–Enfield series of rifles.",
            "Lee–Enfield is often referred to by the acronym of SMLE."
        ],
        "answer": "SMLE"
    },
    {
        "question": "Mister Magoo's Christmas Carol was produced by the same studio that produced a film that featured the only animated-film role by who?",
        "cot": [
            "Mister Magoo's Christmas Carol was produced by United Productions of America studio.",
            "United Productions of America studio produced a film Gay Purr-ee, which features the voice of Judy Garland in her only animated-film role."
        ],
        "answer": "Judy Garland"
    },
    {
        "question": "How many awards did the 'A Girl Like Me' singer win at the American Music Awards of 2012?",
        "cot": [
            "The singer of 'A Girl Like Me' is Rihanna.",
            "In the American Music Awards of 2012, Rihanna won one award."
        ],
        "answer": "one"
    },
    {
        "question": "Which of the following had a debut album entitled 'We Have an Emergency': Hot Hot Heat or The Operation M.D.?",
        "cot": [
            "The debut album of the band 'Hot Hot Heat' was 'Make Up the Breakdown'.",
            "The debut album of the band 'The Operation M.D.' was 'We Have an Emergency'."
        ],
        "answer": "The Operation M.D."
    },
    {
        "question": "Which American actor and director appeared on the series 'December Bride' and in the film 'Gentle Annie'?",
        "cot": [
            "'December Bride' has an American actor and director named Harry Morgan.",
            "The film 'Gentle Annie' also had Harry Morgan as an actor."
        ],
        "answer": "Harry Morgan"
    },
    {
        "question": "Who was born first? Jan de Bont or Raoul Walsh?",
        "cot": [
            "Jan de Bont was born on 22 October 1943.",
            "Raoul Walsh was born on March 11, 1887.",
            "Thus, Raoul Walsh was born first."
        ],
        "answer": "Raoul Walsh"
    }
]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def load_data(self, data_path=None):
        if data_path is None:
            data_path = self.default_data_path
        with open(data_path, "r") as fin:
            js = json.load(fin)
        return js