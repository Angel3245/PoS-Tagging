# Copyright (C) 2024  Jose Ángel Pérez Garrido
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from conllu import parse

class Conllu_parser(object):

    def __init__(self, max_sequence_length=128):
        self.max_sequence_length = max_sequence_length

    def __call__(self,dataset):
        self.inputs = []
        self.targets = []

        with open(dataset, "r", encoding="utf-8") as file:
            sentences = parse(file.read())

        print("Dataset loaded in memory.")
        print("e.g. sentence number 10:",sentences[10])

        for sentence in sentences:

            # Ignore sentences longer than max_sequence_length (128)
            if (len(sentence) <= self.max_sequence_length):
                # Ignore empty tokens and multiword units
                sentence=sentence.filter(id=lambda x: not '-' in str(x) and not '.' in str(x)) 
                
                # Get a sequence
                input = []
                target = []
                for token in sentence:
                    input.append(token["form"])
                    target.append(token["upos"])

                self.inputs.append(' '.join(input))
                self.targets.append(target)
            
            else:
                sentences.remove(sentence)

        return self.inputs, self.targets